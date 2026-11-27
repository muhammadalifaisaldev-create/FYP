"""Generate notebooks/nilm_pipeline.ipynb (then execute it with nbconvert).

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/nilm_pipeline.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Single-phase NILM, stage 1: event detection → steady-state features → Random Forest

Pipeline: **aggregate power → steady-state change detector → differential features from
steady windows (transition excluded) → appliance-event classifier → evaluation against sub-meters.**

### Decisions (short)
| Decision | Choice | Why |
|---|---|---|
| Dataset | **REDD low_freq**, houses 1–3 | 1 Hz whole-home mains + per-circuit sub-meters (≈3 s), so every aggregate event can be labelled automatically. Downloadable here via a public GitHub mirror; the official REDD, UK-DALE, iAWE, BLUED, PLAID hosts are blocked from this sandbox. 1 Hz real-power steps are also what an ESP32 + CT board can deliver first. |
| Aggregate | mains1 + mains2 | REDD homes are US split-phase; the sum is the whole-home signal, equivalent to one Pakistani single-phase feed. |
| Detector | steady-state segmentation (Hart 1992 style) | Simplest credible method; parameters are physical (W, s) and portable to the ESP32. |
| Ground truth | the same detector run on each sub-meter (7 s window) | Gives labelled ON/OFF/step events with ΔP for every metered appliance. |
| Split | **chronological** per house: first 70 % of valid data = train, last 30 % = test, 10 min buffer | No event from the test period is seen during training or detector tuning. |
| Classifier | Random Forest, `class_weight="balanced"`, 300 trees | Plan's first choice; handles unscaled mixed features; compared with kNN and a majority baseline. |
| Normalisation | none | REDD low_freq has no voltage channel, so 230 V normalisation is not possible here (needed later on our own hardware). Trees do not need feature scaling; kNN uses a train-fitted StandardScaler. |
""")

code("""
import sys, json, subprocess
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from nilm.data import download_redd, load_house
from nilm.events import DetectorParams, detect_events, check_windows_exclude_transition
from nilm.synthetic import make_synthetic
from nilm.groundtruth import GT_PARAMS, ground_truth_events, match_events, detection_scores
from nilm.features import event_features, STEADY_FEATURES, TRANSIENT_FEATURES, ALL_FEATURES
from nilm.pipeline import run_house, cross_house, split_time, tune_detector
from nilm.report import synthetic_check, save_all, summary_tables
from nilm import plots
OUT = ROOT / "results"; PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.precision", 3)
download_redd((1, 2, 3))   # no-op when the files are already present
""")

md("## 1. Detector verification on a synthetic signal with known transitions")
code("""
synth, p_syn, truth, ev_syn = synthetic_check(PLOTS)
print(json.dumps(synth, indent=1))
display(pd.concat([truth[["idx", "delta_p", "trans_len", "name"]].add_prefix("true_"),
                   ev_syn[["idx", "delta_p", "trans_start", "trans_end", "pre_start", "pre_end", "post_start", "post_end"]]
                   .add_prefix("det_")], axis=1))
assert synth["true_found"] == synth["n_true"] and synth["false_events"] == 0
assert synth["events_whose_windows_touch_true_transition"] == 0
display(Image(PLOTS / "synthetic_detector.png"))
""")
code("""
# unit tests (5 seeds, spikes, gaps, thresholds)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT)
print(r.stdout[-400:])
""")

md("## 2. Load REDD house 1 and check the data")
code("""
agg, subs, rep = load_house(1)
print(json.dumps(rep, indent=1))
t = agg.index.to_numpy()
assert np.all(np.diff(t) == 1), "grid must be uniform 1 s"
print("aggregate shape", agg.shape, "| sub-meter shape", subs.shape)
print("NaN fraction per sub-meter column where aggregate is valid:")
print(subs[agg.notna()].isna().mean().round(4).to_dict())
print("sub-meter negative values:", int((subs < 0).sum().sum()))
v = agg.notna() & subs.notna().all(axis=1)
print(f"share of aggregate energy explained by sub-meters: {subs[v].sum(axis=1).sum() / agg[v].sum():.1%}")
""")
code("""
day = agg.loc[t[0] + 86400 * 2: t[0] + 86400 * 3]
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(pd.to_datetime(day.index, unit="s"), day.to_numpy(), lw=0.6, color="#191919")
ax.set(title="House 1 aggregate, one day (UTC)", ylabel="W"); fig.tight_layout(); plt.show()
""")

md("## 3. Ground-truth events from sub-meters, chronological split, detector tuning (train only)")
code("""
gt = ground_truth_events(subs, GT_PARAMS, valid=agg.notna())
print(len(gt), "ground-truth events (|ΔP| >= 30 W, only where the aggregate exists)")
print(gt.appliance.value_counts().to_dict())
t_split = split_time(agg, 0.7)
print("split at", pd.to_datetime(t_split, unit="s"), "UTC (70 % of valid seconds before it)")
best, tuning = tune_detector(agg, gt, t_split, buffer_s=600)
print("chosen detector params (train period only):", best)
display(tuning.head(6))
""")

md("## 4–7. Full pipeline for houses 1–3 (detect → match → features → split → RF → evaluate)")
code("""
results = {h: run_house(h, data=(agg, subs, rep) if h == 1 else None) for h in (1, 2, 3)}
cross = cross_house(results, train_house=1)
det, cls, e2e = save_all(results, cross, synth, OUT)
r1 = results[1]
assert r1["params"] == best
""")

md("### 4. Event detection (aggregate events vs sub-meter events, ±10 s, same sign, similar size)")
code("""
display(det)
print("House 1 test recall per appliance:"); display(r1["per_app_recall"])
print("steady windows exclude the transition on real data:", r1["windows_check"])
ev1 = r1["events"]; m = ev1[ev1.gt_dt.notna()]
print("timing offset det - sub-meter (s) percentiles 5/50/95:", np.percentile(m.gt_dt, [5, 50, 95]))
display(Image(PLOTS / "house1_test_excerpt.png"))
""")

md("### 5. Feature checks")
code("""
f = ev1[ALL_FEATURES]
print("NaN:", int(f.isna().sum().sum()), "| inf:", int(np.isinf(f.to_numpy()).sum()))
display(f.describe().T)
display(Image(PLOTS / "house1_feature_space.png"))
""")

md("### 6. Classification (test period only), class balance")
code("""
display(Image(PLOTS / "class_balance.png"))
for h, r in results.items():
    print(f"House {h} - classes not evaluable (<20 train or <5 test events):",
          r["not_evaluable"].to_dict(orient="index") or "none")
display(cls)
display(e2e)
print(r1["report_txt"])
display(Image(PLOTS / "house1_confusion.png"))
print("RF feature importance (house 1):"); display(r1["feature_importance"].round(3))
print("RF predictions for test detections with NO sub-meter match (not scored):")
print(r1["unmatched_test_predictions"].to_dict())
""")
code("""
for h in (2, 3):
    print(f"House {h}"); print(results[h]["report_txt"])
display(Image(PLOTS / "house2_confusion.png")); display(Image(PLOTS / "house3_confusion.png"))
print("Cross-house (train on all house-1 events, test on other houses, shared classes):")
display(cross)
""")

md("### 7. Leakage checks")
code("""
lk = pd.DataFrame({h: r["leakage"] for h, r in results.items()})
display(lk)
for h, r in results.items():
    L = r["leakage"]
    assert L["no_time_overlap"] and L["no_shared_events"] and L["no_shared_gt_events"]
    assert L["gap_seconds"] >= 2 * 600 - 1
print("all leakage assertions passed")
""")

md("### 8. Example events (right and wrong)")
code("""
ex = pd.read_csv(OUT / "example_events.csv")
display(ex[ex.house == 1])
display(Image(PLOTS / "house1_examples.png"))
""")

md("""
## 9. Findings (from the run above)

See `docs/research_log.md` for the narrative and numbers. In short:

* The detector finds every step in the synthetic test, ignores spikes, sub-threshold blips and gaps, and its
  steady windows never touch transition samples (also checked on all real events).
* On REDD, event F1 on the held-out period is ≈0.76–0.85 per house. Misses are concentrated in the washer/dryer
  (motor noise and overlapping steps) and in busy periods; part of the "false positives" are real events of
  unmetered loads (~24 % of house-1 energy is not sub-metered), so precision is a lower bound.
* On correctly detected test events, Random Forest reaches 80–96 % accuracy (macro-F1 0.71–0.90), well above
  the majority baseline. Transient features (overshoot, transition length) add 4–10 accuracy points over steady-state-only features.
* Models do **not** transfer between homes (house1→house3 macro-F1 ≈ 0.46): per-home training/calibration is required.

**Limitations:** REDD mains are apparent power, sub-meters are real power; no reactive power, voltage or
harmonics at 1 Hz; ground truth is derived from sub-meters with the same detector; mixed circuits
(outlets, lighting) are labels, not single appliances; OFF events of multi-state appliances are classified
independently (no ON-list pairing yet).
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"}})
out = ROOT / "notebooks" / "nilm_pipeline.ipynb"
nbf.write(nb, out)
print("wrote", out)
