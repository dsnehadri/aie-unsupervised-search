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
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
SUF = os.environ.get("LOCKIN_SUFFIX", "120")   # "120" = the 120 s campaigns
# Which campaign feeds each trace. The defaults are the two campaigns behind
# the figure in the paper: the fabric-only t2k image and the hybrid v3x image,
# both remeasured 2026-09-15 (+0.634 W and +1.727 W above idle). The earlier
# 90 s campaigns are not in the repository any more.
#
# WHY THIS IS PARAMETERISED. The old campaigns set the ON window by an ITERATION
# COUNT, so a slower design ran a LONGER window: the PL run held its load for
# 118.1 s while the hybrid held its for 108.4 s. The shaded compute band is the
# shorter of the two, so the PL trace stayed high for ~10 s past the band and
# looked like it was decaying late. It was not: its load really was still
# running. The time-based script fixes this at the source, and the current
# campaigns are 122.1 s and 124.1 s, 2 s apart rather than 10.
PL_TAG  = os.environ.get("LOCKIN_PL_TAG",  SUF + "_pl_t2k")
AIE_TAG = os.environ.get("LOCKIN_AIE_TAG", SUF + "_hyb_v3x")
PL_LABEL  = os.environ.get("LOCKIN_PL_LABEL",  "PL-only")
AIE_LABEL = os.environ.get("LOCKIN_AIE_LABEL", "AIE-PL hybrid")

FIGS = "/home/snehadri/repos/aie-unsupervised-search/figs"
RUNS = [
    (PL_LABEL, PL_LABEL, "#eb6834",
     f"{FIGS}/board_thermal_lockin{PL_TAG}_log.csv",
     f"{FIGS}/board_thermal_lockin{PL_TAG}_phases.txt"),
    (AIE_LABEL, AIE_LABEL, "#2a78d6",
     f"{FIGS}/board_thermal_lockin{AIE_TAG}_log.csv",
     f"{FIGS}/board_thermal_lockin{AIE_TAG}_phases.txt"),
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
    if key == "total_W":   # VCCINT INA226 senses one of six phases and reads negative under
        vals = np.array([float(r["total_W"]) - float(r["VCCINT_W"]) + abs(float(r["VCCINT_W"]))   # phase shedding
                         for r in rows])
    else:
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


def fold_cycles(rows, epochs, ons, period, key):
    """Like fold(), but keep the cycles apart: (bin centres, n_cycles x n_bins),
    NaN where a cycle has no sample in a bin."""
    if key == "total_W":
        vals = np.array([float(r["total_W"]) - float(r["VCCINT_W"]) + abs(float(r["VCCINT_W"]))
                         for r in rows])
    else:
        vals = np.array([float(r[key]) for r in rows])
    nb = int(np.ceil(period / BIN))
    starts = np.array([cs for cs, _ in ons])
    idx = np.searchsorted(starts, epochs, side="right") - 1
    ok = idx >= 0
    off = np.full(len(epochs), np.nan)
    off[ok] = epochs[ok] - starts[idx[ok]]
    m = ok & (off >= 0) & (off < period) & np.isfinite(vals)
    b = np.clip((off[m] / BIN).astype(int), 0, nb - 1)
    out = np.full((len(starts), nb), np.nan)
    sums = np.zeros((len(starts), nb)); cnts = np.zeros((len(starts), nb))
    np.add.at(sums, (idx[m], b), vals[m]); np.add.at(cnts, (idx[m], b), 1)
    np.divide(sums, cnts, out=out, where=cnts > 0)
    return (np.arange(nb) + 0.5) * BIN, out


def roll(xs, k=5):
    """Light centered smoothing only. A 200-cycle fold already averages ~200
    samples per 1 s bin, and a wide window visibly drags the sharp ON->OFF
    step earlier than it occurs (a +/-15 s window moved it 15 s). The folded
    cycle is periodic, so the window wraps rather than truncating at the ends."""
    h = k // 2
    pad = np.concatenate([xs[-h:], xs, xs[:h]])
    return np.array([np.nanmean(pad[i:i + 2 * h + 1]) for i in range(len(xs))])


PRE = float(os.environ.get("LOCKIN_PRE_S", "45"))    # pre-load idle shown before t=0
BASE = float(os.environ.get("LOCKIN_BASE_S", "40"))  # settled tail of OFF used as idle
SMOOTH = int(os.environ.get("LOCKIN_SMOOTH", "5"))   # centred smoothing window, s
# Shade the central BAND% of cycles around the mean (e.g. 90 -> 5th..95th
# percentile across cycles), each cycle measured against its own idle so the
# band shows cycle-to-cycle spread, not the slow room drift. Off by default.
BAND = float(os.environ.get("LOCKIN_BAND", "0"))


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
    if not np.isfinite(hi) or not np.isfinite(lo):
        return on_len
    above = t[y > (hi + lo) / 2]
    if above.size == 0:
        return on_len
    return above.max()

loaded = [(lbl, sh, col) + load(c, ph) for lbl, sh, col, c, ph in RUNS]
on_band = min(compute_window(r, e, o, p, ol) for _, _, _, r, e, o, p, ol in loaded)
# Mark the load window with its two edges only; a filled band would sit under
# the shaded spread and hide it.
for ax in (ax1, ax2):
    for xv in (0, on_band):
        ax.axvline(xv, color="#8a8a8a", lw=1.0, ls="--", zorder=0.5)

for label, short, color, rows, epochs, ons, period, on_len in loaded:
    # Baseline-subtract each run at its own idle level: the images sit 3.5 W
    # apart, which would flatten the ~0.3 W compute step this figure is about.
    # Baseline = the settled tail of the OFF window (last BASE s of the cycle).
    def trace(ax, key, lw, ls, lab):
        t, y = fold(rows, epochs, ons, period, key)
        base = np.nanmean(y[t > period - BASE])
        order, x = unwrap(t, period)
        if BAND > 0:
            tc, yc = fold_cycles(rows, epochs, ons, period, key)
            yc = yc - np.nanmean(yc[:, tc > period - BASE], axis=1, keepdims=True)
            # Smooth each cycle BEFORE taking percentiles. Each 1 s bin holds one
            # raw sample per cycle, so percentiles of raw bins measure the power
            # monitor's sample noise (~+-0.4 W), not how the cycles differ.
            yc = np.array([roll(row, SMOOTH) for row in yc])
            lo_q, hi_q = (100 - BAND) / 2, 100 - (100 - BAND) / 2
            lo = np.nanpercentile(yc, lo_q, axis=0)
            hi = np.nanpercentile(yc, hi_q, axis=0)
            ax.fill_between(x, lo[order], hi[order], color=color,
                            alpha=0.18 if ls == "-" else 0.10, linewidth=0, zorder=1)
        ax.plot(x, (roll(y, SMOOTH) - base)[order], color=color, lw=lw, ls=ls, label=lab)

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
# Name the output after the campaigns, so a run with non-default tags cannot
# silently overwrite the figure built from a different pair.
# The default pair is the one in the paper, so it writes the paper's file; any
# other pair must be named explicitly, so it cannot overwrite that figure.
DEFAULT = (PL_TAG, AIE_TAG) == ("120_pl_t2k", "120_hyb_v3x")
out = os.environ.get("LOCKIN_OUT") or (
    f"{FIGS}/board_lockin_pl_vs_aie" + ("" if DEFAULT else f"_{PL_TAG}_{AIE_TAG}") + ".png")
fig.savefig(out, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
