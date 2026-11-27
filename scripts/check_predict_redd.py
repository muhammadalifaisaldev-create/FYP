"""Check predict.py against REDD sub-meters on held-out data.

    python scripts/check_predict_redd.py              # house 1, needs models/redd_house1.joblib

1. Takes the test period of the house (after the model's training cut-off),
   writes one day of it as a plain ``timestamp,watts`` CSV, and runs the same
   inference that predict.py runs on that CSV (so the CSV loader is tested too).
2. Events: matches predicted events to sub-meter ground-truth events
   (±10 s, same sign, similar size) and reports how many detected events got
   the right appliance and the right ON/OFF.
3. Running-appliance timeline: compares the tracker's ON/OFF state of each
   appliance, second by second, with the sub-meter (ON = sub-meter > 30 W above its standby level).
Writes results/predict_check/.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nilm.data import download_redd, load_house  # noqa: E402
from nilm.groundtruth import GT_PARAMS, ground_truth_events, match_events  # noqa: E402
from nilm.inference import load_aggregate_csv, predict, running_at, save_outputs  # noqa: E402
from nilm.model import load_model  # noqa: E402

ON_W = 30.0


def true_on(sub: pd.Series) -> np.ndarray:
    """Sub-meter says ON when it is more than ON_W above its own standby level
    (5th percentile over the period; lighting and outlet circuits never reach 0 W)."""
    base = np.nanpercentile(sub.to_numpy(float), 5) if sub.notna().any() else 0.0
    return (sub.to_numpy(float) > base + ON_W)


def evaluate(agg, subs, bundle, events, intervals):
    classes = bundle["classes"]
    valid = agg.notna()
    gt = ground_truth_events(subs, GT_PARAMS, valid=valid)
    d2g, g2d = match_events(events, gt)
    m = d2g >= 0
    true_app = np.where(m, gt.appliance.to_numpy()[np.maximum(d2g, 0)], None)
    known = m & np.isin(true_app, classes)
    ev = events.assign(true_appliance=np.where(m, true_app, "(no sub-meter event)"))
    ev_scores = {
        "n_predicted_events": int(len(events)), "n_true_events": int(len(gt)),
        "n_true_events_of_model_classes": int(gt.appliance.isin(classes).sum()),
        "detected_and_matched": int(m.sum()),
        "detection_precision": float(m.mean()) if len(m) else 0.0,
        "detection_recall_model_classes": float((g2d[gt.appliance.isin(classes).to_numpy()] >= 0).mean()),
        "on_off_correct_of_matched": float((np.sign(events.delta_p[m]) == np.sign(gt.delta_p.to_numpy()[d2g[m]])).mean()),
        "appliance_accuracy_of_matched_model_classes": float((events.appliance.to_numpy()[known] == true_app[known]).mean()),
        "n_matched_model_classes": int(known.sum()),
    }
    grid = agg.index.to_numpy()
    pred_on = running_at(intervals, grid).reindex(columns=classes, fill_value=False)
    v = valid.to_numpy()
    rows = []
    for c in classes:
        if c not in subs:
            continue
        truth = true_on(subs[c]) & v
        pr = pred_on[c].to_numpy() & v
        tp, fp, fn = int((truth & pr).sum()), int((~truth & pr & v).sum()), int((truth & ~pr).sum())
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        rows.append({"appliance": c, "true_on_min": round(truth.sum() / 60, 1), "pred_on_min": round(pr.sum() / 60, 1),
                     "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else float("nan"),
                     "state_accuracy": float((truth == pr)[v].mean())})
    state = pd.DataFrame(rows)
    # how often the whole running set (model classes) is exactly right
    tr = np.column_stack([true_on(subs[c]) for c in classes if c in subs])
    pd_ = pred_on[[c for c in classes if c in subs]].to_numpy()
    ev_scores["seconds_whole_set_correct"] = float((tr == pd_).all(axis=1)[v].mean())
    always_off = float((~tr).all(axis=1)[v].mean())
    ev_scores["baseline_everything_off_set_correct"] = always_off
    return ev, ev_scores, state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--house", type=int, default=1)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--day", default="2011-05-23", help="held-out day written as the example CSV")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "predict_check")
    a = ap.parse_args(argv)
    bundle = load_model(a.model or ROOT / "models" / f"redd_house{a.house}.joblib")
    assert bundle["source"]["house"] == a.house
    download_redd((a.house,))
    agg, subs, _ = load_house(a.house)
    t_test = bundle["source"]["test_starts_unix"] + 600       # same buffer as training
    assert t_test > bundle["source"]["trained_before_unix"]
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    summary = {}

    # 1) one held-out day through the CSV path, exactly like predict.py
    day0 = int(pd.Timestamp(a.day).timestamp())
    sl = (agg.index >= max(day0, t_test)) & (agg.index < day0 + 86400)
    csv = a.out / f"redd_house{a.house}_{a.day}_aggregate.csv"
    pd.DataFrame({"timestamp": pd.to_datetime(agg.index[sl], unit="s"),
                  "watts": agg[sl].round(2).to_numpy()}).dropna().to_csv(csv, index=False)
    day = load_aggregate_csv(csv)
    ev, tl, iv = predict(day, bundle)
    save_outputs(a.out / a.day, ev, tl, iv)
    ev_day, s_day, st_day = evaluate(day, subs.reindex(day.index), bundle, ev, iv)
    ev_day.drop(columns=["t"]).to_csv(a.out / a.day / "events_vs_truth.csv", index=False)
    st_day.to_csv(a.out / a.day / "state_vs_truth.csv", index=False)
    summary[a.day] = s_day
    print(f"== {a.day} (held-out), CSV {csv.name} ==\n", json.dumps(s_day, indent=1))
    print(st_day.round(3).to_string(index=False))

    # 2) the whole held-out period
    test = agg[agg.index >= t_test]
    ev, tl, iv = predict(test, bundle)
    ev_all, s_all, st_all = evaluate(test, subs.reindex(test.index), bundle, ev, iv)
    st_all.to_csv(a.out / "test_period_state_vs_truth.csv", index=False)
    ev_all.drop(columns=["t"]).to_csv(a.out / "test_period_events_vs_truth.csv", index=False)
    summary["whole_test_period"] = s_all
    print("\n== whole held-out period ==\n", json.dumps(s_all, indent=1))
    print(st_all.round(3).to_string(index=False))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
