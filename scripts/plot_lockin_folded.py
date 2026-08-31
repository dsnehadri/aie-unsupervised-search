#!/usr/bin/env python3
"""Folded ON/OFF load cycle (lock-in), plotted with the same configuration
as plot_board_power_temp.py: board power on top, die temperatures below,
white background, black text, no title, shaded load-ON region.

Two hundred ON/OFF cycles are folded onto one common cycle axis and averaged,
which cancels ambient drift; only light smoothing is then applied.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/snehadri/repos/aie-unsupervised-search/figs/board_thermal_lockin_log.csv"
PHASES = "/home/snehadri/repos/aie-unsupervised-search/figs/board_thermal_lockin_phases.txt"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLACK, GRID = "#1a1a1a", "#dddddd"
BIN = 1.0        # s per folded bin (matches the raw sampling cadence)

rows = list(csv.DictReader(open(CSV)))
epochs = np.array([float(r["epoch"]) for r in rows])

ons = []
for line in open(PHASES):
    p = line.split()
    if p[0] == "on_start":
        ons.append([float(p[1]), None])
    elif p[0] == "on_end" and ons:
        ons[-1][1] = float(p[1])
ons = [(s, e) for s, e in ons if e is not None]
period = np.median([ons[i + 1][0] - ons[i][0] for i in range(len(ons) - 1)])
on_len = np.median([e - s for s, e in ons])

def fold(key):
    """Mean over cycles, per folded time bin."""
    vals = np.array([float(r[key]) for r in rows])
    nb = int(np.ceil(period / BIN))
    acc = [[] for _ in range(nb)]
    for cs, _ in ons:
        m = (epochs >= cs) & (epochs < cs + period)
        for t, v in zip(epochs[m] - cs, vals[m]):
            if np.isfinite(v):
                acc[min(int(t / BIN), nb - 1)].append(v)
    centers = (np.arange(nb) + 0.5) * BIN
    mean = np.array([np.mean(a) if a else np.nan for a in acc])
    return centers, mean

def roll(xs, k=5):
    """Light centered smoothing only. Folding 200 cycles already averages ~200
    samples into every 1 s bin, so heavy smoothing buys nothing and actively
    misleads: a centered +/-15 s window drags the sharp ON->OFF step 15 s
    earlier than it happens, making power look like it sags before the load
    stops. The raw fold is flat until the ON window ends."""
    h = k // 2
    return [np.nanmean(xs[max(0, i - h):min(len(xs), i + h + 1)])
            for i in range(len(xs))]

def col(key):
    c, m = fold(key)
    return c, roll(m)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.6), dpi=150, sharex=True,
                               gridspec_kw={"hspace": 0.12})
for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BLACK)
    ax.tick_params(colors=BLACK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axvspan(0, on_len, color=GRID, alpha=0.45, zorder=0)

# --- power panel ---
t, p = col("total_W")
ax1.plot(t, p, color=BLUE, lw=2, label="AIE-PL hybrid image")
ax1.set_ylabel("Board power (W)", color=BLACK, fontsize=10)
ax1.legend(frameon=False, fontsize=9, labelcolor=BLACK, loc="center right")

# --- temperature panel ---
t, v = col("versal")
ax2.plot(t, v, color=BLUE, lw=2, label="Versal die")
t, a = col("aie")
ax2.plot(t, a, color=BLUE, lw=1.4, ls="--", label="AIE array")
ax2.set_ylabel("Temperature (°C)", color=BLACK, fontsize=10)
ax2.set_xlabel("Time within cycle (s)", color=BLACK, fontsize=10)
ax2.legend(frameon=False, fontsize=8.5, labelcolor=BLACK, ncol=2,
           loc="lower center")

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/board_lockin_folded_cycle.png"
fig.savefig(out, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
