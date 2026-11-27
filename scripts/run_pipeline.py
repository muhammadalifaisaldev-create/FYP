"""Run the full pipeline on REDD houses 1-3 and write results/ .

    python scripts/run_pipeline.py            # downloads REDD low_freq if missing

Also saves the trained models to models/redd_house<N>.joblib for scripts/predict.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from nilm.data import download_redd  # noqa: E402
from nilm.model import bundle_from_result, save_model  # noqa: E402
from nilm.pipeline import cross_house, run_house  # noqa: E402
from nilm.report import save_all, synthetic_check  # noqa: E402

HOUSES = (1, 2, 3)


def main(out=ROOT / "results"):
    download_redd(HOUSES)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    synth, *_ = synthetic_check(out / "plots")
    print("synthetic detector test:", synth)
    results = {h: run_house(h) for h in HOUSES}
    for h, r in results.items():   # trained models for scripts/predict.py
        print("saved", save_model(bundle_from_result(r), ROOT / "models" / f"redd_house{h}.joblib"))
    cross = cross_house(results, train_house=1)
    det, cls, e2e = save_all(results, cross, synth, out)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(det, cls, e2e, cross, sep="\n\n")
        for h, r in results.items():
            print(f"\nHouse {h}\n", r["report_txt"], "\nleakage:", r["leakage"])


if __name__ == "__main__":
    main()
