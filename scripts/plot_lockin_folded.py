#!/usr/bin/env python3
"""Folded ON/OFF load cycle (lock-in): board power over die temperatures.

Paper style, matching plot_board_power_temp.py: white background, black
text, no title, capitalized labels. Cycles are folded onto one common
cycle axis and averaged; the shaded band is +/-1 standard error across
repetitions, so the band -- not the sample-to-sample jitter -- shows the
precision of the measurement.
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
BIN = 2.0        # s per folded bin

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
print(f"{len(ons)} cycles, period {period:.0f} s, ON {on_len:.0f} s")

def fold(key):
    """Return (bin centers, mean, standard error) folded over cycles."""
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
    sem = np.array([np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
                    for a in acc])
    return centers, mean, sem

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

def band(ax, key, color, label):
    c, m, e = fold(key)
    ax.plot(c, m, color=color, lw=1.8, label=label, zorder=3)
    ax.fill_between(c, m - e, m + e, color=color, alpha=0.25, lw=0, zorder=2)
    return c, m, e

# --- power ---
c, m, e = band(ax1, "total_W", BLUE, "Board power")
ax1.set_ylabel("Board power (W)", color=BLACK, fontsize=10)
on = (c > 35) & (c < on_len)
off = c > on_len + 20
dP = np.nanmean(m[on]) - np.nanmean(m[off])
sP = np.sqrt(np.nanmean(e[on] ** 2) / on.sum() + np.nanmean(e[off] ** 2) / off.sum())
ax1.set_ylim(10.55, 11.52)   # headroom so the step is centered

# --- temperatures ---
band(ax2, "versal", BLUE, "Versal die")
band(ax2, "aie", ORANGE, "AIE array")
ax2.set_ylabel("Temperature (°C)", color=BLACK, fontsize=10)
ax2.set_xlabel("Time within cycle (s)", color=BLACK, fontsize=10)
ax2.set_ylim(34.02, None)
ax2.legend(frameon=False, fontsize=9, labelcolor=BLACK, ncol=2, loc="lower center")

c2, m2, e2 = fold("versal")
on2 = (c2 > 35) & (c2 < on_len)
off2 = c2 > on_len + 20
dT = np.nanmean(m2[on2]) - np.nanmean(m2[off2])
print(f"ON-OFF power delta:  {dP*1000:+.0f} mW +/- {sP*1000:.0f}")
print(f"ON-OFF Versal dT:    {dT:+.3f} C")

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/board_lockin_folded_cycle.png"
fig.savefig(out, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
