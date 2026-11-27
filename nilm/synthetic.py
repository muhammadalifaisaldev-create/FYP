"""Synthetic 1 Hz aggregate signal with known ON/OFF transitions, used to
verify the event detector."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(seed: int = 0, n: int = 1800, noise_w: float = 2.0):
    """Return (p, truth) where truth lists every true step with the indices of
    its transition samples (the samples that are neither old nor new level)."""
    rng = np.random.default_rng(seed)
    p = np.full(n, 120.0)                      # base load
    truth = []

    def step(t0, dp, ramp=0, overshoot=0.0, name=""):
        # ramp samples t0 .. t0+ramp-1 are intermediate; new level from t0+ramp
        if ramp:
            frac = (np.arange(1, ramp + 1)) / (ramp + 1)
            p[t0:t0 + ramp] += dp * frac
        p[t0 + ramp:] += dp
        if overshoot:
            p[t0 + ramp] += overshoot      # one-sample inrush spike
            ramp += 1
        truth.append({"idx": t0, "delta_p": dp, "trans_len": ramp,
                      "trans_idx": list(range(t0, t0 + ramp)), "name": name})

    step(100, 150, name="fridge ON")
    step(220, 1500, ramp=2, name="kettle ON")
    step(300, -1500, ramp=1, name="kettle OFF")
    step(420, 60, name="lamp ON")
    step(600, 1100, ramp=1, overshoot=600, name="microwave ON")
    step(780, -1100, name="microwave OFF")
    step(900, -150, ramp=1, name="fridge OFF")
    step(1200, 40, name="small ON")
    step(1400, -60, name="lamp OFF")
    step(1500, -40, name="small OFF")
    p += rng.normal(0, noise_w, n)
    # distractors that must NOT produce events
    p[500] += 400                 # isolated 1-sample spike, level returns
    p[1000:1003] += 15            # sub-threshold blip
    p[1300:1310] = np.nan         # data gap
    return p, pd.DataFrame(truth)
