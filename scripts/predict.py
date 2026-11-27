"""Tell which appliances switched ON/OFF and which are running, from aggregate power.

    python scripts/predict.py my_power.csv --model models/redd_house1.joblib
    python scripts/predict.py my_power.csv --model models/redd_house1.joblib --out results/my_house

Input CSV: one row per reading, ~1 Hz.
    timestamp,watts
    2011-05-20 06:00:00,212.4            (or unix seconds: 1305871200,212.4)
The first column is time and the second is power unless --time-col/--power-col
are given. Train a model first with scripts/train_model.py (or run_pipeline.py).

Prints the event table and the ON intervals; with --out also writes
events.csv, timeline.csv and on_intervals.csv.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from nilm.inference import load_aggregate_csv, predict, save_outputs  # noqa: E402
from nilm.model import load_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="aggregate power CSV (time, watts)")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "redd_house1.joblib")
    ap.add_argument("--out", type=Path, default=None, help="folder for events/timeline/on_intervals CSVs")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--power-col", default=None)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="label events below this classifier probability as 'unknown' (e.g. 0.5)")
    ap.add_argument("--max-rows", type=int, default=40, help="rows to print per table (0 = all)")
    a = ap.parse_args(argv)

    bundle = load_model(a.model)
    agg = load_aggregate_csv(a.csv, a.time_col, a.power_col)
    events, timeline, intervals = predict(agg, bundle, a.min_confidence)

    src = bundle.get("source", {})
    print(f"model: {a.model.name} (REDD house {src.get('house', '?')}), classes: {', '.join(bundle['classes'])}")
    print(f"input: {len(agg)} s from {pd.to_datetime(agg.index[0], unit='s')} to "
          f"{pd.to_datetime(agg.index[-1], unit='s')}, {int(agg.isna().sum())} s missing")
    print(f"{len(events)} events, {len(intervals)} ON periods\n")
    n = None if a.max_rows == 0 else a.max_rows
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 70):
        print("EVENTS")
        print(timeline.drop(columns=["t"]).head(n).to_string(index=False))
        print("\nON PERIODS (off = NaT: still running at the end of the file)")
        print(intervals.drop(columns=["on_t", "off_t"]).head(n).to_string(index=False))
    if a.out:
        paths = save_outputs(a.out, events, timeline, intervals)
        print("\nwrote", ", ".join(str(p) for p in paths.values()))
    return events, timeline, intervals


if __name__ == "__main__":
    main()
