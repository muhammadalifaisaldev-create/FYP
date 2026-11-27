"""Ground-truth appliance events from sub-meters, and matching of detected
aggregate events to them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import DetectorParams, detect_events

# Sub-meters are clean (one load per channel), so the same detector is used,
# with a looser relative tolerance for motor/heater drift and a 7 s steady
# window: sub-meters report every ~3 s and are forward-filled onto the 1 s
# grid, so a 3-sample window would accept a single reading (e.g. an inrush
# spike) as a steady state.
GT_PARAMS = DetectorParams(window=7, tol_abs=10.0, tol_rel=0.05, min_delta=30.0,
                           feat_len=5, max_transition=30)


def ground_truth_events(subs: pd.DataFrame, prm: DetectorParams = GT_PARAMS,
                        valid: pd.Series | None = None) -> pd.DataFrame:
    """``valid``: boolean mask of seconds where the aggregate exists. Sub-meter
    data outside it is blanked so no ground-truth event is counted where the
    aggregate detector cannot possibly see it (REDD has long mains gaps)."""
    if valid is not None:
        subs = subs.where(np.repeat(valid.to_numpy()[:, None], subs.shape[1], axis=1), np.nan)
    out = []
    t = subs.index.to_numpy()
    for name in subs.columns:
        ev = detect_events(subs[name].to_numpy(), prm, timestamps=t)
        if len(ev):
            out.append(pd.DataFrame({"t": ev.t.to_numpy(), "delta_p": ev.delta_p.to_numpy(),
                                     "appliance": name}))
    if not out:
        return pd.DataFrame(columns=["t", "delta_p", "appliance"])
    return pd.concat(out).sort_values("t").reset_index(drop=True)


def match_events(det: pd.DataFrame, gt: pd.DataFrame, tol_s: float = 10.0,
                 max_ratio: float = 2.5, mag_abs: float = 30.0):
    """Greedy one-to-one matching (closest in time first).

    A detected and a true event can match if they are within ``tol_s``
    seconds, have the same sign, and similar size: either
    |dP_det - dP_gt| <= mag_abs, or 1/max_ratio <= |dP_det|/|dP_gt| <= max_ratio.
    The ratio is loose on purpose: REDD mains are apparent power while
    sub-meters are real power, and some 240 V loads (house 1 dryer) are
    sub-metered on one leg only, so the sub-meter shows about half the step. Returns arrays: gt index for every detected
    event (-1 if unmatched) and det index for every gt event (-1 if missed).
    """
    td, dpd = det.t.to_numpy(float), det.delta_p.to_numpy(float)
    tg, dpg = gt.t.to_numpy(float), gt.delta_p.to_numpy(float)
    order = np.argsort(tg)
    tg_s = tg[order]
    pairs = []
    lo = np.searchsorted(tg_s, td - tol_s, "left")
    hi = np.searchsorted(tg_s, td + tol_s, "right")
    for i in range(len(td)):
        for k in range(lo[i], hi[i]):
            j = order[k]
            if np.sign(dpd[i]) != np.sign(dpg[j]):
                continue
            ratio = abs(dpd[i]) / max(abs(dpg[j]), 1e-9)
            if abs(dpd[i] - dpg[j]) > mag_abs and not (1 / max_ratio <= ratio <= max_ratio):
                continue
            pairs.append((abs(td[i] - tg[j]), abs(dpd[i] - dpg[j]), i, j))
    pairs.sort()
    d2g = np.full(len(td), -1)
    g2d = np.full(len(tg), -1)
    for _, _, i, j in pairs:
        if d2g[i] < 0 and g2d[j] < 0:
            d2g[i], g2d[j] = j, i
    return d2g, g2d


def detection_scores(d2g, g2d) -> dict:
    tp = int((d2g >= 0).sum())
    fp = int((d2g < 0).sum())
    fn = int((g2d < 0).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
