"""Prompt-selection dynamics figure from bandit_dynamics.py output (run locally).

(a) Share of the four organs' bandits choosing box, point or box+point in each block:
faint marks per block, lines a 10-block moving average. (b) Organ-mean block reward, raw
and 10-block moving average. The dashed line ends the round-robin warm-up.

    python fig_dynamics.py dyn_seed42.json fig_dynamics.pdf
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

d = json.load(open(sys.argv[1]))
ARMS = (("box", "Box", "#2a78d6"), ("point", "Point", "#eb6834"), ("both", "Box+point", "#7b4fb8"))
INK, INK2, GRID, WARM, REW = "#0b0b0b", "#52514e", "#e6e5e1", "#c0392b", "#2e8b57"
SMOOTH = 10

blocks = d["blocks"]
x = np.array([b["block"] for b in blocks])
share = {a: np.array([sum(v == a for v in b["arms"].values()) / len(b["arms"]) for b in blocks]) for a, _, _ in ARMS}
reward = np.array([np.mean(list(b["rewards"].values())) for b in blocks])


def moving(v):
    """Trailing moving average (shorter window at the start)."""
    c = np.cumsum(np.insert(v, 0, 0.0))
    n = np.minimum(np.arange(1, len(v) + 1), SMOOTH)
    return (c[1:] - c[np.arange(1, len(v) + 1) - n]) / n


plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True)
for a, lab, col in ARMS:
    ax1.scatter(x, share[a], s=2, color=col, alpha=0.15, linewidths=0, zorder=2)
    ax1.plot(x, moving(share[a]), color=col, lw=1.2, label=lab, zorder=3)
ax1.set_ylabel("Selection share", fontsize=6.6)
ax1.set_ylim(-0.03, 1.03)
ax2.plot(x, reward, color=REW, lw=0.6, alpha=0.35, label="Raw", zorder=2)
ax2.plot(x, moving(reward), color=REW, lw=1.3, label=f"{SMOOTH}-block mean", zorder=3)
ax2.set_ylabel("Mean block reward", fontsize=6.6)
ax2.set_xlabel("Decision block", fontsize=6.6)
for ax, tag in ((ax1, "(a) Prompt selection"), (ax2, "(b) Block reward")):
    ax.axvline(d["warmup_blocks"] + 0.5, color=WARM, lw=0.8, ls=(0, (3, 2)), label="Warm-up end", zorder=4)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(tag, fontsize=6.8, loc="left", pad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, ncol=4 if ax is ax1 else 3,
              handlelength=1.8, columnspacing=0.8)
ax1.legend(loc="upper right", frameon=False, fontsize=5.6, ncol=4, handlelength=1.8, columnspacing=0.8)
ax2.set_xlim(0, x[-1] + 1)
fig.tight_layout(h_pad=0.6)
fig.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.02)
print("saved", sys.argv[2], "blocks", len(blocks))
