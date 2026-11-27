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
| `nilm/model.py`, `nilm/inference.py` | save/load a trained model; inference on new aggregate power + ON-state tracker |
| `scripts/train_model.py` | train one house and save `models/redd_house<N>.joblib` |
| `scripts/predict.py` | **your aggregate CSV → events + running appliances** |
| `scripts/check_predict_redd.py` | checks `predict.py` against sub-meters on held-out REDD data |
| `notebooks/predict_example.ipynb` | the same inference as a short executed notebook |
| `tests/test_events.py` | detector unit tests on synthetic signals |
| `results/` | metrics (`metrics.json`, CSVs) and `results/plots/*.png` |
| `docs/research_log.md` | method, decisions, actual findings |

### Setup and run
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+ (tested on 3.13)
pip install -r requirements.txt
python -m pytest -q tests                              # 18 tests (detector + inference)
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

### Using your own aggregate data
You only need a CSV of whole-home power at about 1 Hz and a trained model file:
```bash
python scripts/train_model.py --house 1                # once: trains on REDD house 1, saves models/redd_house1.joblib
python scripts/predict.py my_power.csv --model models/redd_house1.joblib --out results/my_house
```
`my_power.csv` (first column time, second column watts; other names via `--time-col/--power-col`):
```
timestamp,watts
2011-05-23 00:00:00,1596.15
2011-05-23 00:00:01,1598.68
```
Time can be a date-time string or unix seconds. The data is put on a 1 s grid (gaps up to 2 s are filled,
longer gaps stay missing and no event is reported across them). Then the model's own detector settings,
the same features and the Random Forest are applied, and you get:

* **events** (`events.csv`): time, ΔP, ON (ΔP > 0) or OFF (ΔP < 0), predicted appliance, confidence;
* **timeline** (`timeline.csv`): after each event, which appliances are running and their estimated total W;
* **ON periods** (`on_intervals.csv`): appliance, on, off, duration, estimated power.

`--min-confidence 0.5` labels uncertain events `unknown`. The running-appliance tracker keeps an ON list:
an ON adds the appliance, an OFF removes it (an OFF predicted as an appliance that is not on is given to
the ON appliance with the closest power), an appliance is dropped if the meter reading after an event is too
low for it to still be on, or if it stays on longer than it ever did in training, and the list is cleared
after a data gap. The same thing in Python: `notebooks/predict_example.ipynb`.

**Important:** a model only knows the appliances of the home it was trained on (cross-house macro-F1 was
0.46–0.57). A REDD model will run on your file but will name your appliances after REDD house 1's. For your
own home you need labelled events from that home (e.g. a session where each appliance is switched on/off
and noted, or a plug-in reference meter such as the PZEM-004T), which is the next step for the board.

**Check on held-out REDD house 1** (`python scripts/check_predict_redd.py`, data after the training cut-off,
compared with the sub-meters; full tables in `results/predict_check/`):

| | 2011-05-23 (one day) | whole held-out period |
|---|---|---|
| events predicted / true sub-meter events | 207 / 235 | 861 / 998 |
| detected events that match a true event | 90 % | 92 % |
| true events detected | 80 % | 79 % |
| ON/OFF correct (matched events) | 100 % | 100 % |
| appliance correct (matched events) | 81 % | 83 % |

Per appliance, seconds where the tracker's ON/OFF agrees with the sub-meter (ON = > 30 W above standby):
refrigerator F1 0.68, washer/dryer 0.82, bathroom GFI 0.73, dishwasher 0.57, microwave 0.55 over the
held-out period. The timeline is the weak part: always-on circuits (lighting, kitchen outlets) are
rarely seen switching from fully off, so their recall is low (F1 0.16–0.22), and one missed or
misclassified OFF can keep an appliance "on" until the meter-level or time-out rule drops it.

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
* ON/OFF tracking is a simple ON-list (see above), not a full state model (FHMM/HMM); no energy estimation yet.
* The tracker does not know which appliances were already on when the file starts.
