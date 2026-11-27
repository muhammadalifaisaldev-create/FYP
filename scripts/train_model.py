"""Train the appliance-event model on a REDD house and save it for predict.py.

    python scripts/train_model.py --house 1                 # -> models/redd_house1.joblib

The model is fitted on the first 70 % of the house's data only (the same
chronological split as run_pipeline.py), so the last 30 % can be used to
check predict.py against the sub-meters.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nilm.data import download_redd  # noqa: E402
from nilm.model import bundle_from_result, save_model  # noqa: E402
from nilm.pipeline import run_house  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--house", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default: models/redd_house<N>.joblib")
    a = ap.parse_args(argv)
    download_redd((a.house,))
    r = run_house(a.house)
    path = save_model(bundle_from_result(r), a.out or ROOT / "models" / f"redd_house{a.house}.joblib")
    print(f"saved {path}  classes={list(r['model'].classes_)}  "
          f"test accuracy={r['model_scores']['RF steady+transient']['accuracy']:.3f}")
    return r, path


if __name__ == "__main__":
    main()
