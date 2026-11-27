# FYP
NILM
Here you go.

## Stage 1: event detection, steady-state features and appliance-event classification

A tested Python pipeline for single-phase Non-Intrusive Load Monitoring (NILM):

```
aggregate power (1 Hz) ─► steady-state change detector ─► ΔP features from steady windows
                      (transition excluded) ─► Random Forest ─► evaluation vs sub-meters
```

It runs on public data now and is designed so that 1 Hz real-power samples from our ESP32 board can
replace the dataset later (same detector parameters in watts and seconds).

### Dataset
**REDD low_freq** (Kolter & Johnson, 2011), houses 1–3: whole-home mains at 1 Hz (two legs, summed) and
per-circuit sub-meters at ≈3 s. Sub-meters give the ground-truth label of every aggregate event.
The original REDD server and the UK-DALE / iAWE / BLUED / PLAID hosts are not reachable from our build
sandbox, so the unmodified `low_freq` files are fetched from a public GitHub mirror
(`wuhanstudio/redd-lowfreq`). REDD is for research use; cite Kolter & Johnson (2011) in the report.

### Layout
| Path | What |
|---|---|
| `nilm/data.py` | download + load REDD onto a 1 s grid, data-quality report |
| `nilm/events.py` | steady-state segmentation and event detector, window-exclusion check |
| `nilm/synthetic.py` | synthetic signal with known ON/OFF transitions |
| `nilm/groundtruth.py` | sub-meter ground-truth events, event matching, detection P/R/F1 |
| `nilm/features.py` | steady-state differential + transient features |
| `nilm/pipeline.py` | per-house experiment: chronological split, detector tuning (train only), RF/kNN/baseline, leakage checks, cross-house test |
| `nilm/plots.py`, `nilm/report.py` | figures, metric tables, `results/` writer |
| `notebooks/nilm_pipeline.ipynb` | the complete, executed pipeline with outputs |
| `scripts/run_pipeline.py` | same pipeline from the command line, writes `results/` |
| `scripts/build_notebook.py` | regenerates the notebook |
| `tests/test_events.py` | detector unit tests on synthetic signals |
| `results/` | metrics (`metrics.json`, CSVs) and `results/plots/*.png` |
| `docs/research_log.md` | method, decisions, actual findings |

### Setup and run
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+ (tested on 3.13)
pip install -r requirements.txt
python -m pytest -q tests                              # 13 detector tests
python scripts/run_pipeline.py                         # downloads REDD (~520 MB) to data/redd, writes results/ (~1 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/nilm_pipeline.ipynb   # or open it in Jupyter
```
`data/` is git-ignored. To use another copy of REDD low_freq, put `house_N/channel_M.dat` + `labels.dat` under `data/redd/`.

### Results (held-out last 30 % of each house, from the executed run)
| House | Event precision | Event recall | Event F1 | RF accuracy | RF macro-F1 | Majority baseline acc. |
|---|---|---|---|---|---|---|
| 1 | 0.915 | 0.790 | 0.848 | 0.834 | 0.767 | 0.401 |
| 2 | 0.850 | 0.845 | 0.847 | 0.959 | 0.902 | 0.449 |
| 3 | 0.836 | 0.701 | 0.763 | 0.802 | 0.707 | 0.395 |

Details, per-class reports, confusion matrices and examples: `results/` and the notebook.

### Limitations
* REDD mains are apparent power (VA), sub-meters real power (W); no voltage, reactive power or harmonics at
  1 Hz, so 230 V normalisation and P–Q / harmonic features from the project plan are not yet used.
* Ground truth is produced by running the detector on sub-meters; ~24 % of house-1 energy is unmetered, so
  some "false positives" are real events of unmetered loads (precision is a lower bound).
* Event matching is deliberately tolerant in size (factor 2.5) because some 240 V loads are sub-metered on one
  leg only (house-1 dryer shows half the aggregate step).
* Mixed circuits (kitchen outlets, lighting, "outlets_unknown") are labels, not single appliances.
* Classes with < 20 train or < 5 test events are not evaluated (listed in the notebook).
* Models are per-home; a house-1 model does not transfer to other homes.
* No ON/OFF pairing (ON-list tracking) or energy estimation yet.
