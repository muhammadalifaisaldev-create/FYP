"""Generate notebooks/predict_example.ipynb (then execute it with nbconvert).

    python scripts/build_predict_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/predict_example.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = [
    nbf.v4.new_markdown_cell("""# Using your own aggregate data: which appliances are running?

Input: a CSV of whole-home power at ~1 Hz (`timestamp,watts`). Output: every step event with its ΔP,
predicted appliance and ON/OFF, plus the ON periods of each appliance.

The example file is a **held-out** day of REDD house 1 (after the model's training period), written as a
plain CSV by `scripts/check_predict_redd.py`. Replace `CSV` with your own file. The model must be trained
on the same home (`python scripts/train_model.py --house 1` for REDD)."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from nilm.model import load_model
from nilm.inference import load_aggregate_csv, predict, save_outputs
pd.set_option("display.width", 160)

CSV = ROOT / "results/predict_check/redd_house1_2011-05-23_aggregate.csv"   # <- your file here
MODEL = ROOT / "models/redd_house1.joblib"

bundle = load_model(MODEL)
agg = load_aggregate_csv(CSV)                      # 1 s grid, unix-second index
events, timeline, on_periods = predict(agg, bundle)
print(len(events), "events,", len(on_periods), "ON periods")
timeline.drop(columns="t").head(15)"""),
    nbf.v4.new_code_cell("""on_periods.drop(columns=["on_t", "off_t"]).head(15)"""),
    nbf.v4.new_code_cell("""# what is running at a given moment
at = pd.Timestamp("2011-05-23 18:30")
print(timeline[timeline.time <= at].iloc[-1][["time", "running", "running_est_w"]])"""),
    nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, height_ratios=[2, 1.4])
ax1.plot(pd.to_datetime(agg.index, unit="s"), agg.to_numpy(), lw=0.6, color="0.3")
ax1.set_ylabel("aggregate (W)")
apps = sorted(on_periods.appliance.unique())
for i, a in enumerate(apps):
    g = on_periods[on_periods.appliance == a]
    off = g.off.fillna(pd.to_datetime(agg.index[-1], unit="s"))
    ax2.hlines([i] * len(g), g.on, off, lw=6)
ax2.set_yticks(range(len(apps)), apps)
ax2.set_title("predicted ON periods")
plt.tight_layout()"""),
    nbf.v4.new_markdown_cell("""Check against the sub-meters (held-out data): `python scripts/check_predict_redd.py` writes
`results/predict_check/summary.json` and the per-appliance tables; see the README for the numbers."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"}})
out = ROOT / "notebooks" / "predict_example.ipynb"
nbf.write(nb, out)
print("wrote", out)
