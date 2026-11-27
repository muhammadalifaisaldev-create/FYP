"""Inference on new aggregate power: which appliances switched, and which are running.

    aggregate CSV ─► 1 s grid ─► detect_events ─► event_features ─► classifier
                 ─► event table (time, ΔP, appliance, ON/OFF) ─► ON-state tracker
                 ─► running-appliances timeline + ON intervals

Only the trained model file is needed; the detector settings come from it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .events import detect_events
from .features import event_features
from .model import detector_params

UNKNOWN = "unknown"


# --------------------------------------------------------------------------- input
def load_aggregate_csv(path, time_col: str | None = None, power_col: str | None = None,
                       max_ffill: int = 2) -> pd.Series:
    """Read a CSV of aggregate power and put it on a 1 s grid.

    The time column may be unix seconds (int/float) or a date-time string
    (e.g. ``2011-04-18 13:22:05``). By default the first column is time and
    the second is power in watts. Readings in the same second are averaged;
    gaps of up to ``max_ffill`` seconds are forward-filled, longer gaps stay
    NaN (no event is reported across them).

    Returns a float Series indexed by integer unix seconds.
    """
    df = pd.read_csv(path)
    time_col = time_col or df.columns[0]
    power_col = power_col or df.columns[1]
    t = df[time_col]
    if pd.api.types.is_numeric_dtype(t):
        sec = np.floor(t.to_numpy(float)).astype(np.int64)
    else:
        ts = pd.to_datetime(t, utc=False)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        sec = (ts.astype("datetime64[s]").astype(np.int64)).to_numpy()
    s = pd.Series(pd.to_numeric(df[power_col], errors="coerce").to_numpy(float), index=sec)
    s = s.groupby(level=0).mean().sort_index()
    grid = np.arange(s.index.min(), s.index.max() + 1)
    return s.reindex(grid).ffill(limit=max_ffill).rename("watts")


# --------------------------------------------------------------------------- events
def classify_events(agg: pd.Series, bundle: dict, min_confidence: float = 0.0) -> pd.DataFrame:
    """Detect and classify every step event in ``agg`` (1 s grid, unix-second index).

    ``state`` is ON for a rise (ΔP > 0) and OFF for a drop. Events whose
    classifier confidence is below ``min_confidence`` are labelled ``unknown``.
    """
    p = agg.to_numpy(float)
    t = agg.index.to_numpy()
    ev = detect_events(p, detector_params(bundle), timestamps=t)
    cols = ["time", "t", "delta_p", "state", "appliance", "confidence", "pre_w", "post_w"]
    if len(ev) == 0:
        return pd.DataFrame(columns=cols)
    X = event_features(p, ev)[bundle["features"]]
    model = bundle["model"]
    proba = model.predict_proba(X)
    best = proba.argmax(axis=1)
    conf = proba[np.arange(len(best)), best]
    app = np.asarray(model.classes_, dtype=object)[best]
    app[conf < min_confidence] = UNKNOWN
    out = pd.DataFrame({
        "time": pd.to_datetime(ev.t.to_numpy(), unit="s"),
        "t": ev.t.to_numpy(),
        "delta_p": ev.delta_p.round(1).to_numpy(),
        "state": np.where(ev.delta_p.to_numpy() > 0, "ON", "OFF"),
        "appliance": app,
        "confidence": conf.round(2),
        "pre_w": ev.pre_med.round(1).to_numpy(),
        "post_w": ev.post_med.round(1).to_numpy(),
    })
    return out[cols]


# --------------------------------------------------------------------------- tracker
def track_states(events: pd.DataFrame, min_delta: float = 30.0, max_ratio: float = 2.5,
                 gaps: list[tuple[int, int]] | None = None, max_on_s: dict | None = None):
    """Simple ON-list tracker.

    Keeps the estimated power of every appliance that is ON.

    * ON event (ΔP > 0): the predicted appliance is ON; its power += ΔP
      (a second ON step of a multi-state appliance adds to it).
    * OFF event (ΔP < 0): applied to the predicted appliance if it is ON and
      its power is compatible with |ΔP| (ratio within ``max_ratio``);
      otherwise (a classifier mistake) to the ON appliance whose power is
      closest to |ΔP|, if compatible; otherwise the OFF is ignored.
      An appliance is switched OFF when its remaining power < ``min_delta``.
    * Consistency with the meter: the appliances that are ON cannot draw more
      than the aggregate itself. After every event, while the ON list's total
      exceeds the measured level after the event (``post_w``) by more than
      max(50 W, 10 %), the appliance whose power best explains the excess is
      switched OFF (its OFF step was missed or misclassified).
    * Timeout: an appliance ON for longer than ``max_on_s[appliance]``
      seconds (longest ON period seen in training) is switched OFF at that
      time, since its OFF step must have been missed.
    * A data gap (``gaps``: list of (start, end) unix seconds) clears the
      ON list, because steps inside the gap were not seen.
    * The state before the first event is unknown: loads that are already on
      when the file starts (and never switch) are not reported.

    Returns (timeline, intervals):
      timeline : one row per event with the appliance it was applied to
                 (``tracked_as``) and the set of appliances running after it.
      intervals: one row per ON period: appliance, on, off, duration, power.
    """
    on: dict[str, float] = {}
    since: dict[str, int] = {}
    peak: dict[str, float] = {}
    rows, spans = [], []
    gaps = sorted(gaps or [])
    gi = 0

    def close(a, t_off):
        spans.append({"appliance": a, "on_t": since[a], "off_t": t_off, "est_power_w": round(peak[a], 1)})
        del on[a], since[a], peak[a]

    def compatible(a, dp):
        r = abs(dp) / max(on[a], 1e-9)
        return 1 / max_ratio <= r <= max_ratio

    for e in events.sort_values("t").itertuples():
        while gi < len(gaps) and gaps[gi][0] < e.t:
            for a in list(on):
                close(a, gaps[gi][0])
            gi += 1
        for x in list(on):
            lim = (max_on_s or {}).get(x)
            if lim and e.t - since[x] > lim:
                close(x, since[x] + lim)
        a, dp, applied, forced = e.appliance, float(e.delta_p), None, []
        if dp > 0:
            applied = a
            if a not in on:
                on[a], since[a], peak[a] = 0.0, int(e.t), 0.0
            on[a] += dp
            peak[a] = max(peak[a], on[a])
        else:
            if a in on and compatible(a, dp):
                applied = a
            elif on:
                cand = min(on, key=lambda x: abs(on[x] - abs(dp)))
                if compatible(cand, dp):
                    applied = cand
            if applied is not None:
                on[applied] += dp
                if on[applied] < min_delta:
                    close(applied, int(e.t))
        level = getattr(e, "post_w", None)
        if level is not None and not pd.isna(level):
            while on and sum(on.values()) > level + max(50.0, 0.1 * abs(level)):
                excess = sum(on.values()) - level
                x = min(on, key=lambda k: abs(on[k] - excess))
                forced.append(x)
                close(x, int(e.t))
        rows.append({"time": e.time, "t": int(e.t), "delta_p": dp, "state": e.state,
                     "predicted": a, "confidence": getattr(e, "confidence", None),
                     "tracked_as": applied or "(ignored)",
                     "forced_off": ", ".join(forced),
                     "running": ", ".join(sorted(on)) or "-",
                     "running_est_w": round(sum(on.values()), 1)})
    for a in list(on):
        spans.append({"appliance": a, "on_t": since[a], "off_t": None, "est_power_w": round(peak[a], 1)})
    timeline = pd.DataFrame(rows, columns=["time", "t", "delta_p", "state", "predicted", "confidence",
                                           "tracked_as", "forced_off", "running", "running_est_w"])
    iv = pd.DataFrame(spans, columns=["appliance", "on_t", "off_t", "est_power_w"])
    if len(iv):
        iv["on"] = pd.to_datetime(iv.on_t, unit="s")
        iv["off"] = pd.to_datetime(iv.off_t, unit="s")
        iv["duration_min"] = ((iv.off_t - iv.on_t) / 60).round(1)
        iv = iv.sort_values("on_t").reset_index(drop=True)
    else:
        iv = iv.assign(on=[], off=[], duration_min=[])
    return timeline, iv[["appliance", "on", "off", "duration_min", "est_power_w", "on_t", "off_t"]]


def find_gaps(agg: pd.Series, min_gap_s: int = 60) -> list[tuple[int, int]]:
    """(start, end) unix seconds of NaN runs longer than ``min_gap_s``."""
    na = agg.isna().to_numpy()
    if not na.any():
        return []
    d = np.diff(np.r_[0, na.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    t = agg.index.to_numpy()
    return [(int(t[s]), int(t[e - 1])) for s, e in zip(starts, ends) if e - s >= min_gap_s]


def running_at(intervals: pd.DataFrame, t_grid: np.ndarray) -> pd.DataFrame:
    """Boolean ON/OFF matrix (seconds × appliances) from tracker intervals."""
    out = {}
    end = int(t_grid[-1]) + 1
    for a, g in intervals.groupby("appliance"):
        m = np.zeros(len(t_grid), bool)
        for on_t, off_t in zip(g.on_t, g.off_t):
            off = end if off_t is None or pd.isna(off_t) else int(off_t)
            m |= (t_grid >= on_t) & (t_grid < off)
        out[a] = m
    return pd.DataFrame(out, index=t_grid)


def predict(agg: pd.Series, bundle: dict, min_confidence: float = 0.0):
    """Full inference: (events, timeline, intervals)."""
    ev = classify_events(agg, bundle, min_confidence)
    tl, iv = track_states(ev, min_delta=bundle["detector"].get("min_delta", 30.0), gaps=find_gaps(agg),
                          max_on_s=bundle.get("max_on_s"))
    return ev, tl, iv


def save_outputs(out_dir, events, timeline, intervals) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"events": out_dir / "events.csv", "timeline": out_dir / "timeline.csv",
             "intervals": out_dir / "on_intervals.csv"}
    events.drop(columns=["t"]).to_csv(paths["events"], index=False)
    timeline.drop(columns=["t"]).to_csv(paths["timeline"], index=False)
    intervals.drop(columns=["on_t", "off_t"]).to_csv(paths["intervals"], index=False)
    return paths
