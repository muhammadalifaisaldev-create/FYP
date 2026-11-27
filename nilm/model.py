"""Save / load a trained appliance-event model.

A model file bundles everything inference needs, so the same detector
settings used in training are applied to new aggregate data:

* the fitted Random Forest and the feature column order,
* the detector parameters (tuned on the training period),
* the appliance classes, their typical step size (median |ΔP| in training)
  and a maximum ON duration (from the training sub-meters), used by the
  ON-state tracker to drop an appliance whose OFF step was missed,
* where the model came from (house, training period end, sklearn version).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .events import DetectorParams
from .features import ALL_FEATURES

FORMAT_VERSION = 1


def on_durations(sub: pd.Series, on_w: float = 30.0) -> np.ndarray:
    """Lengths (s) of the ON runs of one sub-meter: power more than ``on_w``
    above its standby level (5th percentile). Runs touching a gap are dropped."""
    x = sub.to_numpy(float)
    if not np.isfinite(x).any():
        return np.array([])
    on = x > np.nanpercentile(x, 5) + on_w
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    nan = np.isnan(x)
    ok = [(a > 0 and b < len(x) and not nan[a - 1] and not nan[b]) for a, b in zip(s, e)]
    return (e - s)[np.asarray(ok, bool)] if len(s) else np.array([])


def max_on_seconds(subs: pd.DataFrame, classes, quantile: float = 0.98, floor_s: int = 600) -> dict:
    out = {}
    for c in classes:
        if c in subs:
            d = on_durations(subs[c])
            d = d[d >= 10]
            out[c] = int(max(floor_s, np.quantile(d, quantile))) if len(d) else None
    return out


def bundle_from_result(r: dict) -> dict:
    """Build a model bundle from a ``nilm.pipeline.run_house`` result.

    The model in the result is fitted on the *training period only*, so the
    test period stays usable for checking inference."""
    tr = r["train"]
    return {
        "format_version": FORMAT_VERSION,
        "model": r["model"],
        "features": list(ALL_FEATURES),
        "detector": r["params"].to_dict(),
        "classes": [str(c) for c in r["model"].classes_],
        "typical_abs_delta_p": tr.groupby("label").delta_p.apply(lambda s: float(s.abs().median())).to_dict(),
        "max_on_s": max_on_seconds(r["subs"].loc[:int(tr.t_post_end.max())], r["model"].classes_),
        "source": {"house": int(r["house"]), "trained_before_unix": int(tr.t_post_end.max()),
                   "test_starts_unix": int(r["t_split"]), "n_train_events": int(len(tr))},
        "sklearn_version": sklearn.__version__,
    }


def save_model(bundle: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_model(path) -> dict:
    b = joblib.load(path)
    if not isinstance(b, dict) or "model" not in b or "detector" not in b:
        raise ValueError(f"{path} is not a nilm model file (expected a bundle saved by nilm.model.save_model)")
    return b


def detector_params(bundle: dict) -> DetectorParams:
    return DetectorParams(**bundle["detector"])
