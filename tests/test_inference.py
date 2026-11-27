import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from nilm.events import DetectorParams, detect_events
from nilm.features import ALL_FEATURES, event_features
from nilm.inference import load_aggregate_csv, predict, track_states
from nilm.model import load_model, save_model


def _ev(rows):
    df = pd.DataFrame(rows, columns=["t", "delta_p", "appliance", "post_w"])
    df["time"] = pd.to_datetime(df.t, unit="s")
    df["state"] = np.where(df.delta_p > 0, "ON", "OFF")
    return df


def test_tracker_on_off_and_overlap():
    ev = _ev([(10, 1500, "kettle", 1600), (20, 200, "fridge", 1800),
              (100, -1490, "kettle", 310), (200, -205, "fridge", 100)])
    tl, iv = track_states(ev)
    assert tl.running.tolist() == ["kettle", "fridge, kettle", "fridge", "-"]
    assert iv[["appliance", "on_t", "off_t"]].values.tolist() == [["kettle", 10, 100], ["fridge", 20, 200]]


def test_tracker_reassigns_misclassified_off_and_ignores_impossible_off():
    ev = _ev([(10, 1500, "kettle", 1600), (50, -40, "kettle", 1560),     # too small for the kettle: ignored
              (100, -1480, "fridge", 120)])                              # fridge not ON: closest match = kettle
    tl, iv = track_states(ev)
    assert tl.tracked_as.tolist() == ["kettle", "(ignored)", "kettle"]
    assert iv.off_t.tolist() == [100]


def test_tracker_meter_consistency_and_timeout():
    # kettle OFF step missed, the meter level drops to 100 W -> kettle must be off
    ev = _ev([(10, 1500, "kettle", 1600), (100, -60, "lamp", 100)])
    _, iv = track_states(ev)
    assert iv.off_t.tolist() == [100]
    # fridge OFF never seen: switched off after its maximum ON time
    ev = _ev([(0, 200, "fridge", 300), (5000, 60, "lamp", 360)])
    _, iv = track_states(ev, max_on_s={"fridge": 1800})
    assert iv.set_index("appliance").off_t["fridge"] == 1800


def test_load_csv_iso_and_unix(tmp_path):
    t = np.arange(1_300_000_000, 1_300_000_010)
    w = np.r_[np.full(5, 100.0), np.full(5, 300.0)]
    keep = np.r_[0:3, 5:10]                    # 2 s hole -> forward-filled
    pd.DataFrame({"ts": t[keep], "p": w[keep]}).to_csv(tmp_path / "u.csv", index=False)
    pd.DataFrame({"timestamp": pd.to_datetime(t[keep], unit="s"), "watts": w[keep]}).to_csv(tmp_path / "i.csv", index=False)
    a, b = load_aggregate_csv(tmp_path / "u.csv"), load_aggregate_csv(tmp_path / "i.csv")
    assert a.index.tolist() == b.index.tolist() == t.tolist()
    assert a.isna().sum() == 0 and np.allclose(a, b)


def test_predict_end_to_end_with_saved_model(tmp_path):
    rng = np.random.default_rng(0)
    p = np.full(3000, 100.0)
    truth = []
    for start, dur, w, name in [(200, 400, 1500, "kettle"), (900, 900, 200, "fridge"), (2200, 300, 1500, "kettle")]:
        p[start:start + dur] += w
        truth.append(name)
    p += rng.normal(0, 1, len(p))
    prm = DetectorParams()
    ev = detect_events(p, prm)
    y = np.where(ev.delta_p.abs() > 800, "kettle", "fridge")
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(event_features(p, ev)[ALL_FEATURES], y)
    bundle = {"model": rf, "features": ALL_FEATURES, "detector": prm.to_dict(), "classes": list(rf.classes_)}
    path = save_model(bundle, tmp_path / "m.joblib")
    t0 = 1_300_000_000
    pd.DataFrame({"timestamp": pd.to_datetime(t0 + np.arange(len(p)), unit="s"), "watts": p}).to_csv(tmp_path / "a.csv", index=False)
    events, timeline, iv = predict(load_aggregate_csv(tmp_path / "a.csv"), load_model(path))
    assert events.state.tolist() == ["ON", "OFF"] * 3
    assert iv.appliance.tolist() == truth
    assert ((iv.on_t - t0).tolist(), (iv.off_t - t0).tolist()) == ([200, 900, 2200], [600, 1800, 2500])
