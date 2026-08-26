#!/usr/bin/env python3
"""Board power + die temperatures vs time: PL-only vs AIE-PL hybrid image,
aligned at load start. Paper version: white background, black text, no
title. Data: sustained-load sensor logs (INA226 rails + hwmon temps),
preserved in aie_scratch_save_20260810/thermal_runs/."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/snehadri/aie_scratch_save_20260810/thermal_runs"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLACK, GRID = "#1a1a1a", "#dddddd"

def load_run(csv_path, phase_path):
    rows = list(csv.DictReader(open(csv_path)))
    ph = {}
    for line in open(phase_path):
        p = line.split()
        ph[p[0]] = int(p[1])
    t0 = float(rows[0]["epoch"])
    ls = ph["load_start"] - t0
    t = [float(r["epoch"]) - t0 - ls for r in rows]          # 0 = load start
    return rows, t, ph["load_end"] - t0 - ls

def roll(xs, k=30):
    return [sum(xs[max(0, i - k + 1):i + 1]) / (i - max(0, i - k + 1) + 1)
            for i in range(len(xs))]

def col(rows, n):
    return [float(r[n]) for r in rows]

pl_rows, pl_t, pl_end = load_run(f"{DATA}/thermal_log.csv", f"{DATA}/phase_log.txt")
ai_rows, ai_t, ai_end = load_run(f"{DATA}/thermal2_log.csv",
                                 f"{DATA}/phase2_log.txt")
common_end = min(pl_end, ai_end)

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
    ax.axvspan(0, common_end, color=GRID, alpha=0.45, zorder=0)

# --- power panel ---
ax1.plot(pl_t, roll(col(pl_rows, "total_W")), color=BLUE, lw=2,
         label="PL-only image")
ax1.plot(ai_t, roll(col(ai_rows, "total_W")), color=ORANGE, lw=2,
         label="AIE-PL hybrid image")
ax1.set_ylabel("Board power (W)", color=BLACK, fontsize=10)
ax1.legend(frameon=False, fontsize=9, labelcolor=BLACK, loc="center right")

# --- temperature panel ---
ax2.plot(pl_t, roll(col(pl_rows, "versal")), color=BLUE, lw=2,
         label="Versal die, PL-only")
ax2.plot(pl_t, roll(col(pl_rows, "aie")), color=BLUE, lw=1.4, ls="--",
         label="AIE array, PL-only")
ax2.plot(ai_t, roll(col(ai_rows, "versal")), color=ORANGE, lw=2,
         label="Versal die, hybrid")
ax2.plot(ai_t, roll(col(ai_rows, "aie")), color=ORANGE, lw=1.4, ls="--",
         label="AIE array, hybrid")
ax2.set_ylabel("Temperature (°C)", color=BLACK, fontsize=10)
ax2.set_xlabel("Time relative to load start (s)", color=BLACK, fontsize=10)
ax2.legend(frameon=False, fontsize=8.5, labelcolor=BLACK, ncol=2, loc="center left")

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/board_pl_vs_aie_power_temp.png"
fig.savefig(out, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
