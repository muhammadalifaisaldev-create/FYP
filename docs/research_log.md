# Research log – NILM stage 1 (2026-11-27)

## Goal
Working, measured baseline: detect appliance switching events in whole-home power, describe each with
steady-state differential features, classify the appliance, evaluate honestly.

## Data
* Tried: REDD official server, UK-DALE (UKERC EDC), iAWE, Harvard Dataverse, Zenodo, Figshare, Kaggle,
  Hugging Face → all blocked from the sandbox. GitHub raw files are reachable.
* Chose **REDD low_freq houses 1–3** from an unmodified GitHub mirror. 1 Hz mains, ≈3 s sub-meters.
* House 1: 36 days of wall time, 1.58 M valid seconds (50 % of the span is gaps), 0 negative values,
  sub-meters missing for 0.06 % of valid seconds, sub-meters explain 76 % of aggregate energy.

## Method
1. **Grid**: mains1+mains2 on a 1 s grid (forward-fill ≤ 2 s, else NaN). Sub-meters forward-filled ≤ 8 s;
   channels with the same label summed.
2. **Detector** (`nilm/events.py`): a sample is steady when a `window`-sample window around it has
   range ≤ max(`tol_abs`, `tol_rel`·P) and no jump to its neighbour exceeds that tolerance. Steady
   segments are separated by transitions; consecutive segments whose medians differ by ≥ 30 W give
   an event. No event across NaN gaps; transitions longer than 30 s are ignored.
3. **Ground truth**: same detector on each sub-meter, with a 7 s window (a 3 s window accepted single
   forward-filled readings, e.g. inrush spikes, as steady states and created spurious events), only where
   the aggregate exists.
4. **Matching**: greedy closest-in-time, ±10 s, same sign, similar size.
5. **Features**: steady-state = ΔP, |ΔP|, Δ(std), Δ(std)/|ΔP| from the last/first 5 samples of the
   neighbouring steady segments (never transition samples); transient = transition length, overshoot/|ΔP|.
6. **Split**: chronological per house, 70 % of valid seconds for training, 10 min buffer; an event belongs
   to a side only if its whole footprint does. Detector parameters tuned by grid search (18 settings) on
   the training period only.
7. **Models**: RF (300 trees, balanced class weights), RF on steady-state features only, kNN (k=5,
   scaled), majority baseline.

## Verification (all from executed code)
* Synthetic signal, 10 known steps (30–1500 W, ramps, an inrush spike) + a 1-sample spike, a 15 W blip
  and a 10 s gap: 10/10 found at the exact sample, 0 false events, max ΔP error 3.4 W, 0 events whose
  steady windows touch a true transition sample. 13 pytest tests pass (5 noise seeds).
* Real data: every one of 3 169 house-1 events has pre window < transition < post window.
* Leakage: train/test footprints separated by ≥ 1 500 s in every house, no shared events or shared
  ground-truth events, 0 identical feature rows across splits, tuning and scaling fit on train only.
* Feature table: 0 NaN, 0 inf.

## Findings (held-out test period)
| House | Event P / R / F1 | RF acc. | RF macro-F1 | RF steady-only acc. | kNN acc. | Majority acc. | Detected-and-correct share of GT events |
|---|---|---|---|---|---|---|---|
| 1 | 0.915 / 0.790 / 0.848 | 0.834 | 0.767 | 0.782 | 0.744 | 0.401 | 0.658 |
| 2 | 0.850 / 0.845 / 0.847 | 0.959 | 0.902 | 0.923 | 0.945 | 0.449 | 0.810 |
| 3 | 0.836 / 0.701 / 0.763 | 0.802 | 0.707 | 0.699 | 0.716 | 0.395 | 0.563 |

* **Fridge** is the easiest load (F1 0.90–0.96); microwave 0.83–0.86. Weakest: house-3 dishwasher (0.33),
  oven (0.65) and the mixed circuits in houses 1 and 3 (lighting / outlets 0.61–0.67), and bathroom GFI (precision ≈0.5, confused with
  microwave: both ≈1.5 kW resistive steps).
* The house-1 dishwasher's 225 W pump step is confused with the fridge (similar ΔP): ΔP alone is not enough
  for small motor loads. Transient features add 4–10 accuracy points; reactive power/harmonics from our own
  hardware should add more.
* Detection misses concentrate in the washer/dryer (house-1 recall 0.43: motor noise and overlapping
  steps break steady segments) and in busy periods.
* **Matching tolerance matters**: with a strict size rule (|ΔP_det−ΔP_gt| ≤ max(30 W, 50 %)) house-1 test
  F1 was 0.769 (P 0.831, R 0.715) and RF accuracy 0.849. Inspection showed the house-1 dryer is
  sub-metered on one leg only (2.7 kW on the sub-meter vs 5.2 kW in the mains), so the rule was relaxed to a
  size ratio within ×2.5. Both numbers are reported here so the choice is visible.
* **Cross-house transfer fails**: RF trained on all house-1 events gives macro-F1 0.57 on house 2 and 0.46 on
  house 3 for the shared classes. The final system needs per-home training or calibration.
* Not evaluated (no test events): house 2 stove (42 train), disposal; house 3 furnace (213 train), disposal.

## Next
Pair ON and OFF events with an ON-list tracker (gives appliance runtime and energy) and feed the same
detector from the ESP32 board's 1 Hz P (and Q) stream, with labelled recordings of our own appliances.
