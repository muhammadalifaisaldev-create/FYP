import numpy as np
import pytest

from nilm.events import DetectorParams, detect_events, check_windows_exclude_transition
from nilm.synthetic import make_synthetic


@pytest.mark.parametrize("seed", range(5))
def test_detects_every_known_transition(seed):
    p, truth = make_synthetic(seed=seed)
    ev = detect_events(p)
    assert len(ev) == len(truth), ev
    assert np.array_equal(ev.idx.to_numpy(), truth.idx.to_numpy())
    # step size within 5 % (or 5 W)
    err = np.abs(ev.delta_p.to_numpy() - truth.delta_p.to_numpy())
    assert np.all(err <= np.maximum(5, 0.05 * np.abs(truth.delta_p.to_numpy())))


@pytest.mark.parametrize("seed", range(5))
def test_steady_windows_exclude_transition(seed):
    p, truth = make_synthetic(seed=seed)
    ev = detect_events(p)
    assert all(check_windows_exclude_transition(ev).values())
    for e, t in zip(ev.itertuples(), truth.itertuples()):
        win = set(range(int(e.pre_start), int(e.pre_end) + 1)) | set(range(int(e.post_start), int(e.post_end) + 1))
        assert not win & set(t.trans_idx), (e, t)
        # pre window is entirely before, post window entirely after the true change
        assert e.pre_end < t.idx and e.post_start >= t.idx + t.trans_len


def test_no_event_for_spike_blip_or_gap():
    p, _ = make_synthetic()
    ev = detect_events(p)
    for i in (500, 1000, 1300, 1310):
        assert not ((ev.idx - i).abs() <= 3).any()


def test_no_event_across_nan_gap():
    p = np.r_[np.full(50, 100.0), np.full(5, np.nan), np.full(50, 400.0)]
    assert len(detect_events(p)) == 0


def test_threshold_respected():
    p = np.r_[np.full(50, 100.0), np.full(50, 125.0), np.full(50, 160.0)]
    ev = detect_events(p, DetectorParams(min_delta=30))
    assert list(ev.idx) == [100]
