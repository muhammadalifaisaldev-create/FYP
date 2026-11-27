"""Static evaluation figures (matplotlib)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
PRE_C, POST_C, TRANS_C = "#257de0", "#f47230", "#999999"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 1.4,
})


def class_colors(classes):
    return {c: SERIES[i % len(SERIES)] for i, c in enumerate(sorted(classes))}


def _shade_event(ax, x, e, t0=0):
    e = {k: int(getattr(e, k) if hasattr(e, k) else e[k]) for k in
         ["pre_start", "pre_end", "post_start", "post_end", "trans_start", "trans_end"]}
    e = pd.Series(e)
    ax.axvspan(x[e.pre_start] - 0.5, x[e.pre_end] + 0.5, color=PRE_C, alpha=0.18, lw=0)
    ax.axvspan(x[e.post_start] - 0.5, x[e.post_end] + 0.5, color=POST_C, alpha=0.18, lw=0)
    if e.trans_end >= e.trans_start:
        ax.axvspan(x[e.trans_start] - 0.5, x[e.trans_end] + 0.5, color=TRANS_C, alpha=0.25, lw=0)


def plot_synthetic(p, truth, ev, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(p))
    ax = axes[0]
    ax.plot(x, p, color=TEXT, lw=1)
    for t in truth.idx:
        ax.axvline(t, color=MUTED, ls=":", lw=1)
    ax.scatter(ev.idx, ev.post_med, marker="v", s=40, color=SERIES[1], zorder=3, label="detected event")
    ax.set(title=f"Synthetic test signal: {len(truth)} true steps (dotted), {len(ev)} detected; "
                 "spike @500, 15 W blip @1000, gap @1300 must be ignored",
           ylabel="Power (W)")
    ax.legend(loc="upper right")
    # zoom on the microwave ON (ramp + inrush spike)
    ax = axes[1]
    e = ev.iloc[(ev.idx - 600).abs().argmin()]
    sl = slice(585, 620)
    ax.plot(x[sl], p[sl], color=TEXT, marker="o", ms=3)
    _shade_event(ax, x, e)
    ax.set(title="Zoom on one event: pre window (blue), transition (grey), post window (orange) - "
                 "windows never include transition samples", xlabel="Sample (s)", ylabel="Power (W)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_excerpt(r, path, hours=2.0):
    """Aggregate during the test period with detected events (matched / unmatched)
    and missed ground-truth events."""
    agg, ev, gt = r["agg"], r["events"], r["gt"]
    te = ev[ev.split == "test"]
    # pick the test window with the most ground-truth events
    g = gt[gt.split == "test"]
    span = int(hours * 3600)
    starts = g.t.to_numpy()
    counts = np.searchsorted(starts, starts + span) - np.arange(len(starts))
    t_start = int(starts[np.argmax(counts)]) - 300
    t_end = t_start + span
    s = agg.loc[t_start:t_end]
    tt = pd.to_datetime(s.index, unit="s")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(tt, s.to_numpy(), color=TEXT, lw=0.8, label="aggregate (mains 1 + 2)")
    w = te[(te.t >= t_start) & (te.t <= t_end)]
    m, u = w[w.label != "unmatched"], w[w.label == "unmatched"]
    y = agg.reindex(w.t).to_numpy()
    ax.scatter(pd.to_datetime(m.t, unit="s"), agg.reindex(m.t), s=22, color=SERIES[0], zorder=3,
               label=f"detected, matched to sub-meter ({len(m)})")
    ax.scatter(pd.to_datetime(u.t, unit="s"), agg.reindex(u.t), s=30, marker="x", color=SERIES[7], zorder=3,
               label=f"detected, no sub-meter match ({len(u)})")
    miss = g[(g.t >= t_start) & (g.t <= t_end) & (g.det_idx < 0)]
    ax.scatter(pd.to_datetime(miss.t, unit="s"), agg.reindex(miss.t), s=30, marker="s", facecolor="none",
               edgecolor=SERIES[3], zorder=3, label=f"missed sub-meter event ({len(miss)})")
    ax.set(title=f"House {r['house']} - test-period excerpt ({hours:g} h, UTC)", ylabel="Power (W)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return len(y)


def plot_confusion(r, path):
    cm, classes = r["confusion"], r["evaluable"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(classes) + 1.5, 0.6 * len(classes) + 1.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted", ylabel="True (sub-meter)",
           title=f"House {r['house']} - RF confusion matrix on test events (count, row %)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="row share")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_balance(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8), squeeze=False)
    for ax, (h, r) in zip(axes[0], results.items()):
        c = r["class_counts"].sort_values("train")
        yy = np.arange(len(c))
        ax.barh(yy + 0.2, c.train, height=0.38, color=SERIES[0], label="train")
        ax.barh(yy - 0.2, c.test, height=0.38, color=SERIES[1], label="test")
        labels = [f"{n}{'' if n in r['evaluable'] else ' (not evaluated)'}" for n in c.index]
        ax.set_yticks(yy, labels)
        ax.set(title=f"House {h}: labelled events per class", xlabel="events")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_space(r, path):
    tr = r["train"]
    cols = class_colors(r["evaluable"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in sorted(r["evaluable"]):
        d = tr[tr.label == c]
        axes[0].scatter(d.delta_p, d.delta_std, s=9, alpha=0.6, color=cols[c], label=c, edgecolor="none")
        axes[1].scatter(d.delta_p, d.overshoot_ratio, s=9, alpha=0.6, color=cols[c], label=c, edgecolor="none")
    axes[0].set(xscale="symlog", xlabel="ΔP (W, symlog)", ylabel="Δ std of steady window (W)",
                title=f"House {r['house']} train events: steady-state features")
    axes[1].set(xscale="symlog", xlabel="ΔP (W, symlog)", ylabel="transient overshoot / |ΔP|",
                title="Transient feature", ylim=(-0.1, min(6, tr.overshoot_ratio.quantile(0.99) * 1.2)))
    axes[1].legend(loc="upper left", fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pick_examples(r, seed=0):
    te, ev, gt = r["test"], r["events"], r["gt"]
    rng = np.random.default_rng(seed)

    def take(df, n):
        return df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)] if len(df) else df

    ok = take(te[te.label == te.pred], 3)
    bad = take(te[te.label != te.pred], 3)
    fp = take(ev[(ev.split == "test") & (ev.label == "unmatched")], 1)
    miss = take(gt[(gt.split == "test") & (gt.det_idx < 0) & gt.appliance.isin(r["evaluable"])], 1)
    return ok, bad, fp, miss


def plot_examples(r, path, seed=0):
    agg, subs = r["agg"], r["subs"]
    t0 = int(agg.index[0])
    p = agg.to_numpy()
    ok, bad, fp, miss = pick_examples(r, seed)
    items = ([("correct", e) for _, e in ok.iterrows()] + [("wrong", e) for _, e in bad.iterrows()]
             + [("false pos.", e) for _, e in fp.iterrows()] + [("missed", e) for _, e in miss.iterrows()])
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    rows = []
    for ax, (kind, e) in zip(axes.flat, items):
        if kind == "missed":
            i, app = int(e.t) - t0, e.appliance
        else:
            i = int(e.idx)
            app = e.label if e.label != "unmatched" else None
        sl = slice(max(0, i - 40), i + 60)
        x = np.arange(sl.start, sl.stop) - i
        ax.plot(x, p[sl], color=TEXT, lw=1, label="aggregate")
        if app is not None and app in subs:
            ax.plot(x, subs[app].to_numpy()[sl] + np.nanmin(p[sl]) - np.nanmin(subs[app].to_numpy()[sl]),
                    color=SERIES[2], lw=1, ls="--", label=f"{app} sub-meter (offset)")
        if kind != "missed":
            xx = np.arange(len(p)) - i
            _shade_event(ax, xx, e)
        if kind in ("correct", "wrong"):
            title = f"{kind}: true {e.label}\npred {e.pred}, ΔP={e.delta_p:+.0f} W"
        elif kind == "false pos.":
            title = f"false positive (no sub-meter match)\nΔP={e.delta_p:+.0f} W"
        else:
            title = f"missed: {e.appliance}\nsub-meter ΔP={e.delta_p:+.0f} W"
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("seconds from event")
        ax.legend(fontsize=6.5, loc="best")
        rows.append({"kind": kind, "utc": str(pd.to_datetime(int(e.t), unit="s")),
                     "true": e.get("label", e.get("appliance")) if kind != "missed" else e.appliance,
                     "pred": e.get("pred", "") if kind in ("correct", "wrong") else "",
                     "delta_p_w": round(float(e.delta_p), 1)})
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Power (W)")
    axes[1, 0].set_ylabel("Power (W)")
    fig.suptitle(f"House {r['house']} test-period examples (blue = pre window, grey = transition, "
                 "orange = post window)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return pd.DataFrame(rows)
