"""Make the result figures from results/*.csv and results/examples.json.

Run after run_experiment.py:   python make_figures.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from src.features import N_SLOTS  # noqa: E402
from src.simulate import ROOMS  # noqa: E402

R = Path(__file__).parent / "results"
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NAMES = {"local": "Local only", "central": "Central (raw data shared)",
         "fedavg": "FedAvg", "fedper": "FedPer (personalised)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})

per_home = pd.read_csv(R / "per_home.csv")
cfg = json.loads((R / "config.json").read_text())
n_homes = cfg["n_homes"]


def title(ax, main, sub):
    fig = ax.figure
    fig.text(0.015, 0.975, main, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.905, sub, fontsize=9, color=INK2, va="top")


# ---- Figure 1: overall AUROC per method --------------------------------------
d = per_home[per_home.scoring == "day_mean"]
by_seed = d.groupby(["method", "seed"]).auroc.mean().reset_index()
order = ["fedavg", "local", "central", "fedper"]
stats = by_seed.groupby("method").auroc.agg(["mean", "std"]).loc[order]

fig, ax = plt.subplots(figsize=(7.2, 3.7))
y = np.arange(len(order))
ax.hlines(y, 0.5, stats["mean"], color=GRID, lw=2)
colors = [ORANGE if m == "fedavg" else BLUE for m in order]
ax.errorbar(stats["mean"], y, xerr=stats["std"], fmt="none", ecolor=INK2, lw=1, capsize=3)
ax.scatter(stats["mean"], y, s=70, c=colors, zorder=3, edgecolor=SURFACE, lw=2)
for yi, (m, row) in zip(y, stats.iterrows()):
    ax.text(row["mean"] + row["std"] + 0.006, yi, f"{row['mean']:.3f}", va="center",
            color=INK, fontsize=9)
ax.set_yticks(y, [NAMES[m] for m in order])
ax.set_xlim(0.5, 0.92)
ax.set_xlabel("AUROC on unusual-day detection (higher is better; 0.5 = chance)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
title(ax, "Personalised FL matches central training without sharing raw data",
      f"Mean over {n_homes} simulated homes, {len(cfg['seeds'])} seeds (whisker = std over seeds)")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(F / "fig1_overall_auroc.png", dpi=200)
plt.close(fig)

# ---- Figure 2: FedAvg vs FedPer by home type ----------------------------------
late, sparse = f"H{n_homes - 1:02d}", f"H{n_homes:02d}"
d = d.copy()
d["group"] = np.where(d.home == late, "Late riser\n(sleeps 01:30, wakes 09:50)",
                      np.where(d.home == sparse, "Only 3 sensors\n(no living room or door)",
                               f"Typical homes\n(H01–H{n_homes - 2:02d})"))
groups = [f"Typical homes\n(H01–H{n_homes - 2:02d})",
          "Late riser\n(sleeps 01:30, wakes 09:50)", "Only 3 sensors\n(no living room or door)"]
g = d[d.method.isin(["fedavg", "fedper"])].groupby(["group", "method"]).auroc.mean().unstack()
g = g.loc[groups]

fig, ax = plt.subplots(figsize=(7.2, 3.7))
y = np.arange(len(groups))[::-1]
for yi, (grp, row) in zip(y, g.iterrows()):
    ax.plot([row.fedavg, row.fedper], [yi, yi], color=GRID, lw=3, zorder=1)
ax.scatter(g.fedavg, y, s=70, c=ORANGE, zorder=3, edgecolor=SURFACE, lw=2, label="FedAvg")
ax.scatter(g.fedper, y, s=70, c=BLUE, zorder=3, edgecolor=SURFACE, lw=2,
           label="FedPer (personalised)")
for yi, (grp, row) in zip(y, g.iterrows()):
    lo, hi = sorted([row.fedavg, row.fedper])
    ax.text(row.fedavg + (-0.008 if row.fedavg < row.fedper else 0.008), yi + 0.22,
            f"{row.fedavg:.2f}", ha="right" if row.fedavg < row.fedper else "left",
            color=INK2, fontsize=9)
    ax.text(row.fedper + (0.008 if row.fedper > row.fedavg else -0.008), yi + 0.22,
            f"{row.fedper:.2f}", ha="left" if row.fedper > row.fedavg else "right",
            color=INK, fontsize=9, fontweight="bold")
ax.set_yticks(y, groups)
ax.set_xlim(0.45, 0.95)
ax.set_ylim(-0.6, len(groups) - 0.3)
ax.set_xlabel("AUROC (0.5 = chance)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", fontsize=9)
title(ax, "One shared model fails the late riser; personalisation fixes it",
      "Same data and training budget; only the decoder is kept per home in FedPer")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(F / "fig2_fedavg_vs_fedper_by_home.png", dpi=200)
plt.close(fig)

# ---- Figure 3: worked explanation example --------------------------------------
examples = json.loads((R / "examples.json").read_text())
ex = next((e for e in examples if e["type"] == "night_wandering"), examples[0])
counts = np.array(ex["counts"])
base = np.array(ex["baseline"])
mask = np.array(ex["mask"]).reshape(len(ROOMS), N_SLOTS)[:, 0] > 0
rooms = [r for r, k in zip(ROOMS, mask) if k]
counts, base = counts[mask], base[mask]
seq = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9cc3ef", BLUE, "#0f3f7a"])
div = LinearSegmentedColormap.from_list("div", [ORANGE, "#f4f3ef", BLUE])

fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
vmax = max(counts.max(), base.max())
panels = [(base, "Usual day for this person (mean of their normal days)", seq, None),
          (counts, f"The flagged day (injected: {ex['type'].replace('_', ' ')})", seq, None),
          (counts - base, "Difference: blue = more than usual, orange = less", div,
           TwoSlopeNorm(0, vmin=min(-1, (counts - base).min()), vmax=max(1, (counts - base).max())))]
for ax, (data, label, cmap, norm) in zip(axes, panels):
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                   vmin=None if norm else 0, vmax=None if norm else vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(rooms)), [r.capitalize() for r in rooms])
    ax.set_title(label, loc="left", fontsize=9.5, color=INK, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2)
axes[-1].set_xticks(np.arange(0, N_SLOTS + 1, 6) - 0.5,
                    [f"{h:02d}:00" for h in range(0, 25, 3)])
axes[-1].set_xlabel("Time of day (30-minute slots); colour = sensor events")
fig.suptitle("What the caregiver would read:", x=0.01, ha="left", fontsize=11,
             fontweight="bold", y=0.995)
fig.text(0.01, 0.945, "“" + ex["explanation"][0] + "”", fontsize=9.5, color=INK,
         ha="left", wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(F / "fig3_explanation_example.png", dpi=200)
plt.close(fig)

# ---- Figure 4: privacy noise vs detection --------------------------------------
rows = []
for scoring in ["day_mean", "block_z"]:
    for sigma, m in [(0.0, "fedper")] + [(s, f"fedper_dp{s}") for s in cfg["dp_sigmas"]]:
        v = per_home[(per_home.method == m) & (per_home.scoring == scoring)]
        s_ = v.groupby("seed").auroc.mean()
        rows.append((scoring, sigma, s_.mean(), s_.std()))
p = pd.DataFrame(rows, columns=["scoring", "sigma", "mean", "std"])
fig, ax = plt.subplots(figsize=(7.2, 3.7))
lab = {"day_mean": "Score = average error over the day",
       "block_z": "Score = most unusual 2-hour block (z-score)"}
for scoring, c in [("day_mean", ORANGE), ("block_z", BLUE)]:
    q = p[p.scoring == scoring]
    ax.plot(q.sigma, q["mean"], color=c, lw=2, marker="o", ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=lab[scoring])
    ax.fill_between(q.sigma, q["mean"] - q["std"], q["mean"] + q["std"], color=c, alpha=0.12, lw=0)
    last = q.iloc[-1]
    ax.text(last.sigma + 0.03, last["mean"], f"{last['mean']:.3f}", va="center", color=INK, fontsize=9)
ax.set_xticks([0] + cfg["dp_sigmas"], ["No noise"] + [f"σ = {s}" for s in cfg["dp_sigmas"]])
ax.set_xlim(-0.08, max(cfg["dp_sigmas"]) + 0.2)
ax.set_ylabel("AUROC")
ax.set_ylim(0.65, 0.9)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="lower left")
title(ax, "Privacy noise costs accuracy; the block z-score loses less",
      "FedPer with clipped, noised updates (simplified DP, no formal ε yet). Band = std over seeds")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(F / "fig4_privacy_tradeoff.png", dpi=200)
plt.close(fig)

print("Figures written to", F)
