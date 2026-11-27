"""REDD (low_freq) download and loading.

The official REDD server (redd.csail.mit.edu) is offline / unreachable from
many networks, so by default the files are fetched from a public GitHub mirror
of the unmodified `low_freq` folder. Each channel file is "<unix_ts> <watts>".
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MIRROR = "https://raw.githubusercontent.com/wuhanstudio/redd-lowfreq/master"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "redd"

# Labels that name a single appliance (or a single dedicated circuit).
# Mixed circuits (outlets, lighting, gfi, ...) are kept as their own label.
RENAME = {"dishwaser": "dishwasher", "furance": "furnace"}


def download_redd(houses=(1, 2, 3), dest: Path = DEFAULT_DIR, mirror: str = MIRROR):
    dest = Path(dest)
    for h in houses:
        hd = dest / f"house_{h}"
        hd.mkdir(parents=True, exist_ok=True)
        lab = hd / "labels.dat"
        if not lab.exists():
            urllib.request.urlretrieve(f"{mirror}/house_{h}/labels.dat", lab)
        for line in lab.read_text().split("\n"):
            if not line.strip():
                continue
            ch = int(line.split()[0])
            f = hd / f"channel_{ch}.dat"
            if not f.exists() or f.stat().st_size == 0:
                print("downloading", f)
                urllib.request.urlretrieve(f"{mirror}/house_{h}/channel_{ch}.dat", f)
    return dest


def read_labels(house_dir: Path) -> dict[int, str]:
    out = {}
    for line in Path(house_dir, "labels.dat").read_text().split("\n"):
        if line.strip():
            c, name = line.split()
            out[int(c)] = RENAME.get(name, name)
    return out


def read_channel(house_dir: Path, ch: int) -> pd.Series:
    df = pd.read_csv(Path(house_dir, f"channel_{ch}.dat"), sep=" ", header=None,
                     names=["t", "p"], dtype={"t": np.int64, "p": np.float64})
    df = df.drop_duplicates("t").sort_values("t")
    return pd.Series(df.p.to_numpy(), index=df.t.to_numpy(), name=ch)


def load_house(house: int, root: Path = DEFAULT_DIR, mains_ffill: int = 2,
               sub_ffill: int = 8):
    """Return (agg, subs, report).

    agg  : pd.Series on a 1 s grid (unix seconds) = mains1 + mains2; NaN where
           missing for more than ``mains_ffill`` seconds.
    subs : pd.DataFrame on the same grid, one column per label (channels with
           the same label are summed, e.g. the two legs of a 240 V appliance);
           sub-meters report every ~3 s and are forward-filled up to
           ``sub_ffill`` seconds.
    report: dict of data-quality facts (shapes, gaps, NaNs).
    """
    hd = Path(root) / f"house_{house}"
    labels = read_labels(hd)
    mains = [c for c, n in labels.items() if n == "mains"]
    m = pd.concat([read_channel(hd, c) for c in mains], axis=1, join="inner").sum(axis=1)
    grid = np.arange(m.index.min(), m.index.max() + 1)
    agg = m.reindex(grid).ffill(limit=mains_ffill)
    cols = {}
    for c, n in labels.items():
        if n == "mains":
            continue
        s = read_channel(hd, c).reindex(grid, method="ffill", limit=sub_ffill, tolerance=sub_ffill)
        cols[n] = s if n not in cols else cols[n].add(s, fill_value=0)
        # keep NaN where every contributing channel is missing
    subs = pd.DataFrame(cols, index=grid)
    report = {
        "house": house,
        "mains_channels": mains,
        "n_raw_mains_samples": int(len(m)),
        "grid_seconds": int(len(grid)),
        "valid_agg_seconds": int(agg.notna().sum()),
        "agg_nan_fraction": float(agg.isna().mean()),
        "start_utc": str(pd.to_datetime(grid[0], unit="s")),
        "end_utc": str(pd.to_datetime(grid[-1], unit="s")),
        "n_gaps_over_60s": int((np.diff(m.index) > 60).sum()),
        "labels": sorted(subs.columns),
        "sub_nan_fraction_when_agg_valid": float(subs[agg.notna()].isna().any(axis=1).mean()),
        "agg_min_w": float(agg.min()), "agg_max_w": float(agg.max()),
        "agg_negative_samples": int((agg < 0).sum()),
    }
    return agg, subs, report
