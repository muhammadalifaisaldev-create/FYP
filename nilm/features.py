"""Per-event features.

Steady-state differential features are computed only from the pre/post
steady windows returned by the detector (which exclude the transition).
Transient features are computed only from the transition samples and are
kept as a separate group so their contribution can be measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEADY_FEATURES = ["delta_p", "abs_delta_p", "delta_std", "rel_delta_std"]
TRANSIENT_FEATURES = ["trans_len", "overshoot_ratio"]
ALL_FEATURES = STEADY_FEATURES + TRANSIENT_FEATURES


def event_features(p: np.ndarray, ev: pd.DataFrame) -> pd.DataFrame:
    p = np.asarray(p, float)
    f = pd.DataFrame(index=ev.index)
    f["delta_p"] = ev.delta_p
    f["abs_delta_p"] = ev.delta_p.abs()
    f["delta_std"] = ev.post_std - ev.pre_std           # change in ripple / noise
    f["rel_delta_std"] = f.delta_std / f.abs_delta_p
    f["trans_len"] = ev.trans_len
    over = np.zeros(len(ev))
    for k, (s, e, post, dp) in enumerate(zip(ev.trans_start, ev.trans_end, ev.post_med, ev.delta_p)):
        if e >= s:
            seg = p[s:e + 1]
            ext = seg.max() if dp > 0 else seg.min()
            over[k] = max(0.0, np.sign(dp) * (ext - post)) / abs(dp)
    f["overshoot_ratio"] = over
    return f
