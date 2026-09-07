#!/usr/bin/env python3
"""Folded ON/OFF load cycle (lock-in) for BOTH deployed images, overlaid.

Same plotting configuration as plot_lockin_folded.py / plot_board_power_temp.py:
board power on top, die temperature below, white background, black text, no
title, shaded load-ON region.

Each trace is 200 ON/OFF cycles folded onto one common cycle axis and averaged,
which cancels ambient drift, then the same 30-sample centered rolling mean.
Both campaigns used a matched ~90 s ON / 90 s OFF cadence, so the two folds are
directly comparable.

  PL-only   = BOOT.BIN.plstream_batched2  (true-batched dataflow, 4,869 ev/s)
  AIE-PL   = BOOT.BIN.aie_maskfix        (72-tile hybrid,        8,962 ev/s)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "/home/snehadri/repos/aie-unsupervised-search/figs"
RUNS = [
    ("PL-only  (idle 7.53 W)", "PL-only", "#eb6834",
     f"{FIGS}/board_thermal_lockin_pl_log.csv",
     f"{FIGS}/board_thermal_lockin_pl_phases.txt"),
    ("AIE-PL hybrid  (idle 10.90 W)", "AIE-PL hybrid", "#2a78d6",
     f"{FIGS}/board_thermal_lockin_log.csv",
     f"{FIGS}/board_thermal_lockin_phases.txt"),
]
BLACK, GRID = "#1a1a1a", "#dddddd"
BIN = 1.0


def load(csv_path, phase_path):
    rows = list(csv.DictReader(open(csv_path)))
    epochs = np.array([float(r["epoch"]) for r in rows])
    ons = []
    for line in open(phase_path):
        p = line.split()
        if p[0] == "on_start":
            ons.append([float(p[1]), None])
        elif p[0] == "on_end" and ons:
            ons[-1][1] = float(p[1])
    ons = [(s, e) for s, e in ons if e is not None]
    period = np.median([ons[i + 1][0] - ons[i][0] for i in range(len(ons) - 1)])
    on_len = np.median([e - s for s, e in ons])
    return rows, epochs, ons, period, on_len


def fold(rows, epochs, ons, period, key):
    vals = np.array([float(r[key]) for r in rows])
    nb = int(np.ceil(period / BIN))
    starts = np.array([cs for cs, _ in ons])
    # assign every sample to its cycle, then to a bin within that cycle
    idx = np.searchsorted(starts, epochs, side="right") - 1
    ok = idx >= 0
    off = np.full(len(epochs), np.nan)
    off[ok] = epochs[ok] - starts[idx[ok]]
    m = ok & (off >= 0) & (off < period) & np.isfinite(vals)
    b = np.clip((off[m] / BIN).astype(int), 0, nb - 1)
    sums = np.bincount(b, weights=vals[m], minlength=nb)
    cnts = np.bincount(b, minlength=nb)
    mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return (np.arange(nb) + 0.5) * BIN, mean


def roll(xs, k=5):
    """Light centered smoothing only. A 200-cycle fold already averages ~200
    samples per 1 s bin, and a wide window visibly drags the sharp ON->OFF
    step earlier than it occurs (a +/-15 s window moved it 15 s). The folded
    cycle is periodic, so the window wraps rather than truncating at the ends."""
    h = k // 2
    pad = np.concatenate([xs[-h:], xs, xs[:h]])
    return np.array([np.nanmean(pad[i:i + 2 * h + 1]) for i in range(len(xs))])


PRE = 45.0   # seconds of pre-load idle to show before t=0


def unwrap(t, period):
    """Move the tail of the cycle in front of t=0 so the plot opens on the
    idle baseline: the fold is periodic, so the last PRE seconds of the OFF
    window are exactly the state just before the load starts."""
    x = np.where(t > period - PRE, t - period, t)
    return np.argsort(x), np.sort(x)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.6), dpi=150, sharex=True,
                               gridspec_kw={"hspace": 0.12})
# The shaded band marks the window in which the kernel is actually computing,
# taken from the folded power step (>50% of the step), NOT the host-process
# on_start..on_end window: after the last iteration the host still reads back
# results, prints, and tears down XRT before the shell stamps on_end, so the
# process window runs ~2 s past the point where power falls. Use the shorter of
# the two images' compute windows so neither trace decays inside the band.
def compute_window(rows, epochs, ons, period, on_len):
    t, y = fold(rows, epochs, ons, period, "total_W")
    hi = np.nanmean(y[(t >= 35) & (t < on_len - 5)])
    lo = np.nanmean(y[t > on_len + 40])
    above = t[y > (hi + lo) / 2]
    return above.max()

loaded = [(lbl, sh, col) + load(c, ph) for lbl, sh, col, c, ph in RUNS]
on_band = min(compute_window(r, e, o, p, ol) for _, _, _, r, e, o, p, ol in loaded)
for ax in (ax1, ax2):
    ax.axvspan(0, on_band, color=GRID, alpha=0.45, zorder=0)

for label, short, color, rows, epochs, ons, period, on_len in loaded:
    # Baseline-subtract each run at its own idle level: the images sit 3.5 W
    # apart, which would flatten the ~0.3 W compute step this figure is about.
    # Baseline = the settled tail of the OFF window (last 40 s of the cycle).
    def trace(ax, key, lw, ls, lab):
        t, y = fold(rows, epochs, ons, period, key)
        base = np.nanmean(y[t > period - 40])
        order, x = unwrap(t, period)
        ax.plot(x, (roll(y) - base)[order], color=color, lw=lw, ls=ls, label=lab)

    trace(ax1, "total_W", 2.0, "-", label)
    trace(ax2, "versal", 2.0, "-", f"{short}, Versal die")
    trace(ax2, "aie", 1.4, "--", f"{short}, AIE array")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BLACK)
    ax.tick_params(colors=BLACK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

ax1.set_ylabel("Supply power above idle (W)", color=BLACK, fontsize=10)
ax1.text(0.012, 0.93, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold", va="top", color=BLACK)
# headroom so the top-right legend clears the ON plateau
ax1.set_ylim(top=ax1.get_ylim()[1] * 1.32)
ax1.legend(frameon=False, fontsize=9, labelcolor=BLACK, loc="upper right")
ax2.set_ylabel("Die temperature above idle (°C)", color=BLACK, fontsize=10)
ax2.text(0.012, 0.93, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="top", color=BLACK)
ax2.set_xlabel("Time within cycle (s)", color=BLACK, fontsize=10)
# headroom so the four-entry legend clears the ON plateau and its falling edge
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.65)
ax2.legend(frameon=False, fontsize=8.5, labelcolor=BLACK, ncol=1, loc="upper right")
for ax in (ax1, ax2):
    ax.axhline(0, color=BLACK, lw=0.8, alpha=0.35, zorder=1)

fig.tight_layout()
out = f"{FIGS}/board_lockin_pl_vs_aie.png"
fig.savefig(out, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
