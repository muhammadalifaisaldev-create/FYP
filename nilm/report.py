"""Collect per-house results into tables, save metrics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .events import detect_events, check_windows_exclude_transition
from .synthetic import make_synthetic


def synthetic_check(out_dir: Path | None = None) -> dict:
    p, truth = make_synthetic()
    ev = detect_events(p)
    hit = [int(((ev.idx - t).abs() <= 1).any()) for t in truth.idx]
    in_trans = sum(bool((set(range(int(e.pre_start), int(e.pre_end) + 1))
                         | set(range(int(e.post_start), int(e.post_end) + 1))) & set(tr.trans_idx)) for e, tr in zip(ev.itertuples(), truth.itertuples())) if len(ev) == len(truth) else None
    res = {"n_true": len(truth), "n_detected": len(ev), "true_found": int(sum(hit)),
           "false_events": int(len(ev) - sum(hit)),
           "max_abs_delta_error_w": float(np.max(np.abs(ev.delta_p.to_numpy() - truth.delta_p.to_numpy())))
           if len(ev) == len(truth) else None,
           "events_whose_windows_touch_true_transition": in_trans,
           **check_windows_exclude_transition(ev)}
    if out_dir is not None:
        plots.plot_synthetic(p, truth, ev, Path(out_dir) / "synthetic_detector.png")
    return res, p, truth, ev


def summary_tables(results: dict):
    det = pd.DataFrame([{"house": h, "split": s, **r["det_scores"][s]} for h, r in results.items()
                        for s in ("train", "test")])
    cls = pd.DataFrame([{"house": h, "model": m, **v} for h, r in results.items()
                        for m, v in r["model_scores"].items()])
    e2e = pd.DataFrame([{"house": h, "end_to_end_recall": r["end_to_end_recall"],
                         "n_test_events_classified": len(r["test"]),
                         "classes_evaluated": len(r["evaluable"]),
                         "classes_not_evaluated": ", ".join(f"{c} ({int(row.train)}/{int(row.test)})"
                                                            for c, row in r["not_evaluable"].iterrows()) or "-"}
                        for h, r in results.items()])
    return det, cls, e2e


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    return str(o)


def save_all(results: dict, cross: pd.DataFrame, synth: dict, out_dir: Path):
    out_dir = Path(out_dir)
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    det, cls, e2e = summary_tables(results)
    det.to_csv(out_dir / "event_detection_metrics.csv", index=False)
    cls.to_csv(out_dir / "classification_metrics.csv", index=False)
    e2e.to_csv(out_dir / "end_to_end.csv", index=False)
    cross.to_csv(out_dir / "cross_house.csv", index=False)
    metrics = {"synthetic_detector_test": synth, "houses": {}}
    examples = []
    for h, r in results.items():
        r["per_class"].to_csv(out_dir / f"house{h}_per_class.csv")
        pd.DataFrame(r["confusion"], index=r["evaluable"], columns=r["evaluable"]).to_csv(
            out_dir / f"house{h}_confusion_matrix.csv")
        cols = ["t", "delta_p", "gt_delta_p", "gt_dt", "label", "pred", "abs_delta_p", "delta_std",
                "rel_delta_std", "trans_len", "overshoot_ratio"]
        r["test"][cols].to_csv(out_dir / f"house{h}_test_predictions.csv", index=False)
        if r["tuning"] is not None:
            r["tuning"].to_csv(out_dir / f"house{h}_detector_tuning.csv", index=False)
        plots.plot_confusion(r, pdir / f"house{h}_confusion.png")
        plots.plot_excerpt(r, pdir / f"house{h}_test_excerpt.png")
        ex = plots.plot_examples(r, pdir / f"house{h}_examples.png")
        ex.insert(0, "house", h)
        examples.append(ex)
        metrics["houses"][h] = {
            "data_report": r["report"], "detector_params": r["params"].to_dict(),
            "split_utc": str(pd.to_datetime(r["t_split"], unit="s")),
            "event_detection": r["det_scores"],
            "per_appliance_detection_recall_test": r["per_app_recall"],
            "class_counts": r["class_counts"], "evaluable_classes": r["evaluable"],
            "not_evaluable_classes": r["not_evaluable"], "leakage_checks": r["leakage"],
            "steady_window_check": r["windows_check"], "classification": r["model_scores"],
            "end_to_end_recall": r["end_to_end_recall"],
            "rf_feature_importance": r["feature_importance"],
            "unmatched_test_detections_rf_prediction": r["unmatched_test_predictions"],
        }
    pd.concat(examples).to_csv(out_dir / "example_events.csv", index=False)
    plots.plot_class_balance(results, pdir / "class_balance.png")
    plots.plot_feature_space(results[min(results)], pdir / f"house{min(results)}_feature_space.png")
    metrics["cross_house"] = cross.to_dict(orient="records")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_jsonable))
    return det, cls, e2e
