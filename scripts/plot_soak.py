#!/usr/bin/env python3
"""Steady-state thermal soak: 30 min under continuous load, 20 min idle.

Same look as plot_lockin_folded.py (white background, black text, no title,
shaded load-ON region), but this is a single long cycle rather than a fold of
many short ones, so the point it makes is different: the 120 s lock-in never
reached thermal equilibrium, and this shows where the temperature actually
settles. Absolute values are plotted, not values above idle, because the idle
levels themselves differ between the two images (the AIE array burns a few
watts whether or not it is doing work).

The traces are smoothed with a centred 15 s window: one cycle per design means
one sample per second per point, not the ~50 the fold averaged.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "/home/snehadri/repos/aie-unsupervised-search/figs"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLACK, GRID = "#1a1a1a", "#dddddd"
SMOOTH = 15  # s, centred
PRE = float(os.environ.get("SOAK_PRE", "300"))   # seconds shown before the hold starts


def load(tag):
    rows = list(csv.DictReader(open(f"{FIGS}/board_thermal_soak_{tag}_log.csv")))
    e = np.array([float(r["epoch"]) for r in rows])
    ph = {}
    for line in open(f"{FIGS}/board_thermal_soak_{tag}_phases.txt"):
        p = line.split()
        if len(p) > 1 and p[0] in ("run_start", "on_start", "on_end", "run_end"):
            ph[p[0]] = float(p[1])

    def col(key):
        if key == "total_W":  # VCCINT can read negative under phase shedding
            v = np.array([float(r["total_W"]) - float(r["VCCINT_W"])
                          + abs(float(r["VCCINT_W"])) for r in rows])
        else:
            v = np.array([float(r[key]) for r in rows])
        return v

    t = e - ph["on_start"]
    keep = (t >= -PRE) & (e <= ph["run_end"])
    # The script calibrates its iteration count with a ~30 min load that ends at
    # run_start, so the minutes before the hold are NOT idle: the board is still
    # warm and drawing load power until run_start, then idles for 60 s. Return
    # that boundary so the plot can mark it instead of implying a cold start.
    return (t[keep], {k: col(k)[keep] for k in ("versal", "aie", "total_W")},
            ph["on_end"] - ph["on_start"], ph["run_start"] - ph["on_start"])


def roll(x, k=SMOOTH):
    h = k // 2
    return np.array([np.nanmean(x[max(0, i - h):min(len(x), i + h + 1)])
                     for i in range(len(x))])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.6), dpi=150, sharex=True,
                               gridspec_kw={"hspace": 0.12})
on_len = cal_end = None
for tag, colour, label in (("fabric", ORANGE, "PL-only"),
                           ("hybrid", BLUE, "AIE-PL hybrid")):
    t, d, on_len, cal_end = load(tag)
    ax1.plot(t, roll(d["total_W"]), color=colour, lw=2, label=label)
    ax2.plot(t, roll(d["versal"]), color=colour, lw=2, label=f"{label}, Versal die")
    ax2.plot(t, roll(d["aie"]), color=colour, lw=1.4, ls="--",
             label=f"{label}, AIE array")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BLACK)
    ax.tick_params(colors=BLACK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axvspan(0, on_len, color=GRID, alpha=0.45, zorder=0)
    # Older campaigns calibrated with a full-length load that ended just before
    # the hold, so their trace starts warm; mark that span when it exists. A run
    # with ITERS supplied has no calibration and idles from the reboot.
    if cal_end > -PRE + 1:
        ax.axvspan(-PRE, cal_end, facecolor=GRID, alpha=0.45, hatch="///",
                   edgecolor="#bbbbbb", linewidth=0.0, zorder=0)
    ax.set_xlim(-PRE, None)

if cal_end > -PRE + 1:
    ax1.annotate("calibration\nload", xy=(-PRE / 2, 0.06), xycoords=("data", "axes fraction"),
                 ha="center", fontsize=8, color="#6b6b6b")
ax1.set_ylabel("Supply power (W)", color=BLACK, fontsize=10)
ax1.legend(frameon=False, fontsize=9, labelcolor=BLACK, loc="center right")
ax2.set_ylabel("Die temperature (°C)", color=BLACK, fontsize=10)
ax2.set_xlabel("Time (s)", color=BLACK, fontsize=10)
ax2.legend(frameon=False, fontsize=8.5, labelcolor=BLACK, ncol=2,
           loc="lower center")

fig.tight_layout()
out = f"{FIGS}/board_thermal_soak"
fig.savefig(f"{out}.png", facecolor="white")
fig.savefig(f"{out}.pdf", facecolor="white")
print("saved", out + ".{png,pdf}")
