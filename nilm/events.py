"""Steady-state change (edge) event detector, in the spirit of Hart (1992).

The signal is split into *steady segments*: runs of samples where every
sample is covered by a window of ``window`` samples whose range is within a
tolerance, and no consecutive pair of samples jumps by more than that
tolerance. Anything between two steady segments is the *transition*.
An event is reported when the levels of two consecutive steady segments
differ by at least ``min_delta`` watts.

The steady-state windows used for features are taken strictly inside the
steady segments (last ``feat_len`` samples before the transition and first
``feat_len`` samples after it), so they never contain transition samples.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class DetectorParams:
    window: int = 3            # samples a steady window must span
    tol_abs: float = 10.0      # W, absolute steadiness tolerance
    tol_rel: float = 0.02      # fraction of the local power level
    min_delta: float = 30.0    # W, minimum step size to report
    feat_len: int = 5          # max samples in each steady feature window
    max_transition: int = 30   # max samples allowed between two steady segments

    def to_dict(self):
        return asdict(self)


def _tolerance(p: np.ndarray, prm: DetectorParams) -> np.ndarray:
    return np.maximum(prm.tol_abs, prm.tol_rel * np.abs(p))


def steady_segments(p: np.ndarray, prm: DetectorParams) -> np.ndarray:
    """Return an (n_segments, 2) int array of inclusive [start, end] indices."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    w = prm.window
    if n < w:
        return np.empty((0, 2), dtype=int)
    s = pd.Series(p)
    # window starting at i covers [i, i + w - 1]
    rmax = s.rolling(w).max().shift(-(w - 1)).to_numpy()
    rmin = s.rolling(w).min().shift(-(w - 1)).to_numpy()
    ok_start = (rmax - rmin) <= _tolerance(np.nan_to_num(rmin), prm)
    ok_start &= ~np.isnan(rmax)
    # a sample is steady if some steady window covers it
    covered = np.convolve(ok_start.astype(np.int8), np.ones(w, dtype=np.int8))[:n] > 0
    # a large jump between consecutive samples always separates segments
    d = np.abs(np.diff(p))
    lvl = np.fmin(np.abs(p[:-1]), np.abs(p[1:]))
    jump = np.zeros(n, dtype=bool)
    jump[1:] = ~(d <= _tolerance(np.nan_to_num(lvl), prm))  # NaN diff -> jump
    # run boundaries
    start = covered & (~np.r_[False, covered[:-1]] | jump)
    starts = np.flatnonzero(start)
    if len(starts) == 0:
        return np.empty((0, 2), dtype=int)
    run_id = np.cumsum(start) - 1
    run_id[~covered] = -1
    ends = np.full(len(starts), -1)
    idx = np.flatnonzero(covered)
    np.maximum.at(ends, run_id[idx], idx)
    seg = np.column_stack([starts, ends])
    seg = seg[(seg[:, 1] - seg[:, 0] + 1) >= w]
    return seg


def detect_events(p: np.ndarray, prm: DetectorParams | None = None,
                  timestamps: np.ndarray | None = None) -> pd.DataFrame:
    """Detect step events in a power signal sampled on a uniform grid.

    NaN marks missing data; no event is reported across a NaN gap.
    """
    prm = prm or DetectorParams()
    p = np.asarray(p, dtype=float)
    seg = steady_segments(p, prm)
    cols = ["idx", "pre_start", "pre_end", "post_start", "post_end",
            "trans_start", "trans_end", "trans_len", "pre_med", "post_med",
            "pre_std", "post_std", "delta_p"]
    if len(seg) < 2:
        return pd.DataFrame(columns=cols + (["t"] if timestamps is not None else []))
    a, b = seg[:-1], seg[1:]
    gap = b[:, 0] - a[:, 1] - 1
    # NaN between segments?
    nan_cum = np.r_[0, np.cumsum(np.isnan(p))]
    has_nan = (nan_cum[b[:, 0]] - nan_cum[a[:, 1] + 1]) > 0
    keep = (~has_nan) & (gap <= prm.max_transition)
    a, b, gap = a[keep], b[keep], gap[keep]
    L = prm.feat_len
    pre_s = np.maximum(a[:, 0], a[:, 1] - L + 1)
    pre_e = a[:, 1]
    post_s = b[:, 0]
    post_e = np.minimum(b[:, 1], b[:, 0] + L - 1)
    pre_med = np.array([np.median(p[s:e + 1]) for s, e in zip(pre_s, pre_e)])
    post_med = np.array([np.median(p[s:e + 1]) for s, e in zip(post_s, post_e)])
    pre_std = np.array([np.std(p[s:e + 1]) for s, e in zip(pre_s, pre_e)])
    post_std = np.array([np.std(p[s:e + 1]) for s, e in zip(post_s, post_e)])
    dp = post_med - pre_med
    ev = pd.DataFrame({
        "idx": pre_e + 1, "pre_start": pre_s, "pre_end": pre_e,
        "post_start": post_s, "post_end": post_e,
        "trans_start": pre_e + 1, "trans_end": post_s - 1, "trans_len": gap,
        "pre_med": pre_med, "post_med": post_med,
        "pre_std": pre_std, "post_std": post_std, "delta_p": dp,
    })
    ev = ev[np.abs(ev.delta_p) >= prm.min_delta].reset_index(drop=True)
    if timestamps is not None:
        ev["t"] = np.asarray(timestamps)[ev.idx.to_numpy()]
    return ev


def check_windows_exclude_transition(ev: pd.DataFrame) -> dict:
    """Structural check: pre window ends before the transition, post window
    starts after it, and windows never overlap the transition samples."""
    ok_pre = (ev.pre_end < ev.trans_start).all()
    ok_post = (ev.post_start > ev.trans_end).all()
    ok_order = (ev.pre_start <= ev.pre_end).all() and (ev.post_start <= ev.post_end).all()
    return {"pre_before_transition": bool(ok_pre),
            "post_after_transition": bool(ok_post),
            "windows_well_formed": bool(ok_order),
            "n_events": int(len(ev))}
