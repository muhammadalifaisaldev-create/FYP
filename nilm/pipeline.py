"""End-to-end experiment for one REDD house: detect -> label -> features ->
chronological split -> Random Forest -> evaluation."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_house
from .events import DetectorParams, detect_events, check_windows_exclude_transition
from .features import ALL_FEATURES, STEADY_FEATURES, event_features
from .groundtruth import GT_PARAMS, detection_scores, ground_truth_events, match_events  # noqa: F401

TUNING_GRID = {"window": [3, 5], "tol_abs": [5.0, 10.0, 20.0], "tol_rel": [0.01, 0.02, 0.04]}
MIN_TRAIN, MIN_TEST = 20, 5


def split_time(agg: pd.Series, frac: float = 0.7) -> int:
    """Timestamp at which ``frac`` of the *valid* aggregate seconds lie before."""
    valid_t = agg.index[agg.notna().to_numpy()]
    return int(valid_t[int(frac * len(valid_t))])


def tune_detector(agg, gt, t_split, buffer_s, grid=TUNING_GRID):
    """Grid search on the training period only (aggregate after the split is
    blanked), maximising event F1."""
    p = agg.to_numpy().copy()
    p[agg.index.to_numpy() >= t_split - buffer_s] = np.nan
    gt_tr = gt[gt.t < t_split - buffer_s].reset_index(drop=True)
    rows = []
    for w, ta, tr in itertools.product(grid["window"], grid["tol_abs"], grid["tol_rel"]):
        prm = DetectorParams(window=w, tol_abs=ta, tol_rel=tr)
        ev = detect_events(p, prm, agg.index.to_numpy())
        s = detection_scores(*match_events(ev, gt_tr))
        rows.append({**prm.to_dict(), **s})
    res = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    ints = {"window", "feat_len", "max_transition"}
    best = DetectorParams(**{k: (int if k in ints else float)(res.loc[0, k]) for k in DetectorParams().to_dict()})
    return best, res


def run_house(house: int, frac: float = 0.7, buffer_s: int = 600, seed: int = 0,
              tune: bool = True, data=None):
    agg, subs, report = data if data is not None else load_house(house)
    t = agg.index.to_numpy()
    t0 = int(t[0])
    gt = ground_truth_events(subs, GT_PARAMS, valid=agg.notna())
    t_split = split_time(agg, frac)

    if tune:
        prm, tuning = tune_detector(agg, gt, t_split, buffer_s)
    else:
        prm, tuning = DetectorParams(), None
    ev = detect_events(agg.to_numpy(), prm, t)
    ev["t_pre_start"] = t0 + ev.pre_start
    ev["t_post_end"] = t0 + ev.post_end
    d2g, g2d = match_events(ev, gt)
    ev["gt_idx"] = d2g
    ev["label"] = np.where(d2g >= 0, gt.appliance.to_numpy()[np.maximum(d2g, 0)], "unmatched")
    ev["gt_delta_p"] = np.where(d2g >= 0, gt.delta_p.to_numpy()[np.maximum(d2g, 0)], np.nan)
    ev["gt_dt"] = np.where(d2g >= 0, ev.t - gt.t.to_numpy()[np.maximum(d2g, 0)], np.nan)
    gt["det_idx"] = g2d

    # chronological split with a buffer; an event belongs to a side only if its
    # whole footprint (pre window start .. post window end) is on that side
    ev["split"] = np.select([ev.t_post_end < t_split - buffer_s, ev.t_pre_start > t_split + buffer_s],
                            ["train", "test"], "buffer")
    gt["split"] = np.select([gt.t < t_split - buffer_s, gt.t > t_split + buffer_s],
                            ["train", "test"], "buffer")

    # event-detection scores per split (GT events in the buffer are ignored)
    det_scores = {}
    for s in ("train", "test"):
        g = gt[gt.split == s]
        d = ev[ev.split == s]
        tp = int((d.gt_idx >= 0).sum()); fp = int((d.gt_idx < 0).sum()); fn = int((g.det_idx < 0).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = (len(g) - fn) / len(g) if len(g) else 0.0
        det_scores[s] = {"tp": tp, "fp": fp, "fn": fn, "n_gt": int(len(g)), "precision": P, "recall": R,
                         "f1": 2 * P * R / (P + R) if P + R else 0.0}
    per_app_recall = (gt[gt.split == "test"].assign(hit=lambda d: d.det_idx >= 0)
                      .groupby("appliance").hit.agg(["mean", "size"])
                      .rename(columns={"mean": "recall", "size": "n_gt"}))

    feats = event_features(agg.to_numpy(), ev)
    ev = pd.concat([ev, feats.drop(columns=["delta_p", "trans_len"])], axis=1)

    lab = ev[(ev.label != "unmatched") & (ev.split != "buffer")]
    counts = lab.groupby(["label", "split"]).size().unstack(fill_value=0)
    for c in ("train", "test"):
        if c not in counts:
            counts[c] = 0
    counts = counts[["train", "test"]]
    evaluable = counts[(counts.train >= MIN_TRAIN) & (counts.test >= MIN_TEST)].index.tolist()
    not_evaluable = counts.drop(index=evaluable)
    tr = lab[(lab.split == "train") & lab.label.isin(evaluable)]
    te = lab[(lab.split == "test") & lab.label.isin(evaluable)]

    leakage = {
        "train_last_event_end": int(tr.t_post_end.max()),
        "test_first_event_start": int(te.t_pre_start.min()),
        "gap_seconds": int(te.t_pre_start.min() - tr.t_post_end.max()),
        "no_time_overlap": bool(tr.t_post_end.max() < te.t_pre_start.min()),
        "no_shared_events": len(set(tr.index) & set(te.index)) == 0,
        "no_shared_gt_events": len(set(tr.gt_idx) & set(te.gt_idx)) == 0,
        "detector_tuned_on": "train period only" if tune else "defaults (no tuning)",
        "scaler_and_models_fit_on": "train only",
        "identical_feature_rows_train_test": int(pd.merge(tr[ALL_FEATURES].round(6).drop_duplicates(),
                                                          te[ALL_FEATURES].round(6)).shape[0]),
    }
    windows_check = check_windows_exclude_transition(ev)

    models = {
        "RF steady+transient": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                       random_state=seed, n_jobs=-1), ALL_FEATURES),
        "RF steady-state only": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                        random_state=seed, n_jobs=-1), STEADY_FEATURES),
        "kNN (k=5) steady-state": (make_pipeline(StandardScaler(), KNeighborsClassifier(5)), STEADY_FEATURES),
        "Majority class": (DummyClassifier(strategy="most_frequent"), STEADY_FEATURES),
    }
    model_scores, preds = {}, {}
    for name, (m, cols) in models.items():
        m.fit(tr[cols], tr.label)
        yp = m.predict(te[cols])
        preds[name] = yp
        model_scores[name] = {"accuracy": accuracy_score(te.label, yp),
                              "macro_f1": f1_score(te.label, yp, average="macro"),
                              "weighted_f1": f1_score(te.label, yp, average="weighted")}
    main = "RF steady+transient"
    rf = models[main][0]
    classes = sorted(evaluable)
    te = te.assign(pred=preds[main])
    cm = confusion_matrix(te.label, te.pred, labels=classes)
    report_txt = classification_report(te.label, te.pred, labels=classes, zero_division=0)
    per_class = pd.DataFrame(classification_report(te.label, te.pred, labels=classes, zero_division=0,
                                                   output_dict=True)).T.loc[classes]
    # end-to-end: test GT events of evaluable classes that were detected AND classified correctly
    g_te = gt[(gt.split == "test") & gt.appliance.isin(evaluable)]
    correct_gt = set(te.gt_idx[te.label == te.pred])
    end_to_end = len(correct_gt & set(g_te.index)) / len(g_te) if len(g_te) else 0.0
    # what the model says about unmatched test detections (no ground truth -> not scored)
    um = ev[(ev.split == "test") & (ev.label == "unmatched")]
    um_pred = pd.Series(rf.predict(um[ALL_FEATURES]) if len(um) else [], dtype=object).value_counts()

    return {
        "house": house, "report": report, "params": prm, "tuning": tuning, "t_split": t_split,
        "agg": agg, "subs": subs, "gt": gt, "events": ev, "test": te, "train": tr,
        "det_scores": det_scores, "per_app_recall": per_app_recall,
        "class_counts": counts, "evaluable": classes, "not_evaluable": not_evaluable,
        "leakage": leakage, "windows_check": windows_check, "model_scores": model_scores,
        "confusion": cm, "report_txt": report_txt, "per_class": per_class,
        "end_to_end_recall": end_to_end, "unmatched_test_predictions": um_pred,
        "feature_importance": pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False),
        "model": rf,
    }


def cross_house(results: dict, train_house: int = 1, seed: int = 0) -> pd.DataFrame:
    """Train on every labelled event of one house, test on other houses, using
    only classes evaluable in both. Shows how much a per-home model relies on
    that home's specific appliances."""
    src = results[train_house]
    tr = src["events"][src["events"].label != "unmatched"]
    rows = []
    for h, r in results.items():
        if h == train_house:
            continue
        te = r["events"][r["events"].label != "unmatched"]
        shared = sorted(set(src["evaluable"]) & set(r["evaluable"]))
        a, b = tr[tr.label.isin(shared)], te[te.label.isin(shared)]
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        m.fit(a[ALL_FEATURES], a.label)
        yp = m.predict(b[ALL_FEATURES])
        rows.append({"train_house": train_house, "test_house": h, "classes": ", ".join(shared),
                     "n_train": len(a), "n_test": len(b), "accuracy": accuracy_score(b.label, yp),
                     "macro_f1": f1_score(b.label, yp, average="macro")})
    return pd.DataFrame(rows)
