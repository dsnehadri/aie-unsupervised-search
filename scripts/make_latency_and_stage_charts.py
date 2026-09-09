#!/usr/bin/env python
"""Two figures from one data set: latency_batch_sweep = steady-state invocation time vs batch size for both designs,
whose slopes are the per-event intervals; (b),(c) per-stage time for one event
in each design, with that same interval drawn as the dashed reference line.
Data and conventions identical to make_latency_chart.py and
make_stage_time_chart.py; this only combines them."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

PL_C, AIE_C = "#d62728", "#1f77b4"
PLC, AIEC, AIE_DARK, INK = "#e8d9a0", "#a9cdea", "#5b8fc9", "#1a1a1a"

# ---- (a) batch sweep, steady state (N >= 8) --------------------------------
SWEEP = {
    "PL-only": [(1,0.90110),(2,1.10842),(4,1.51807),(8,2.33875),(16,3.98148),
                (32,7.26646),(64,13.83435),(128,26.96866),(256,53.24013)],
    # cross-event pipelined hybrid with the vector integer layer norm, 2026-09-09.
    # Its FIFO depth no longer matters: shallow and deep both give 57.9 us/event
    # (shallow 0.39223..15.16503 ms). With the earlier float layer norm the same
    # two builds gave 111.0 and 57.9 us -- the deeper FIFOs were compensating for
    # a slow AI Engine stage, and once that stage is fast the buffering is idle.
    "AIE-PL hybrid": [(1,0.39245),(2,0.45408),(4,0.56998),(8,0.80148),(16,1.26358),
                      (32,2.19175),(64,4.04537),(128,7.75214),(256,15.16611)],
}
COL = {"PL-only": PL_C, "AIE-PL hybrid": AIE_C}
NMIN = 8

# ---- (b),(c) per-stage costs, SAME rows in both panels ------------------------
# PL-only: routed cycle counts at 80 MHz. Hybrid: PL parts at 100 MHz from the
# routed report; AIE compute = measured per-block interval (blocks3 sweep, L0
# blocks; L1 uses the same kernels and is taken equal). Hybrid attention rows are
# stacked: light = PL streaming / building on the fabric, dark = AI Engine compute.
PL_CLK, HYB_CLK = 80e6, 100e6
_bi = "/home/snehadri/aie_scratch_save_20260810/block_intervals.json"
_d = json.load(open(_bi)) if os.path.isfile(_bi) else {}
AIE = {k: _d[k]["slope_us"] for k in ("Object attention", "Candidate attention", "Cross attention") if k in _d}
cyc = lambda c, clk: c / clk * 1e6
# rows: (label, PL-only us, hybrid PL-side us, hybrid AIE us)
ROWS = [
 ("Read input",                cyc(149, PL_CLK),   cyc(149, HYB_CLK),                 0),
 ("Fork",                      cyc(153, PL_CLK),   cyc(153, HYB_CLK),                 0),
 ("Embedding",                 cyc(4483, PL_CLK),  cyc(5874, HYB_CLK),                0),
 ("Pairwise $w_{ij}$",         cyc(3028, PL_CLK),  cyc(836, HYB_CLK),                 0),
 ("Object attention L0",       cyc(16269, PL_CLK), cyc(614+241+450, HYB_CLK),         AIE.get("Object attention", 0)),
 ("Build candidates + candidate attention L0", cyc(3499, PL_CLK), cyc(920+64+61, HYB_CLK),      AIE.get("Candidate attention", 0)),
 ("Cross attention L0",        cyc(13102, PL_CLK), cyc(308+241, HYB_CLK),             AIE.get("Cross attention", 0)),
 ("Object attention L1",       cyc(16824, PL_CLK), cyc(267+241+450, HYB_CLK),         AIE.get("Object attention", 0)),
 ("Build candidates + candidate attention L1", cyc(3551, PL_CLK), cyc(920+64+61, HYB_CLK),      AIE.get("Candidate attention", 0)),
 ("Cross attention L1",        cyc(13102, PL_CLK), cyc(308+241, HYB_CLK),             AIE.get("Cross attention", 0)),
 ("Candidate build* + mass",   cyc(747, PL_CLK),   cyc(747, HYB_CLK),                 0),
 ("Autoencoder + MSE",         cyc(792, PL_CLK),   cyc(1162, HYB_CLK),                0),
 ("Write DDR",                 cyc(81, PL_CLK),    cyc(81, HYB_CLK),                  0),
]
labs = [r[0] for r in ROWS]
y = np.arange(len(labs))[::-1]

plt.rcParams.update({"font.size": 11})
figa, axa = plt.subplots(figsize=(7.2, 5.0))
fig, (axb, axc) = plt.subplots(1, 2, figsize=(12.6, 5.6))
fig.subplots_adjust(wspace=0.62)

fits = {}
for name, pts in SWEEP.items():
    n = np.array([a for a, _ in pts], float); t = np.array([b for _, b in pts], float)
    m = n >= NMIN
    slope, icept = np.polyfit(n[m], t[m], 1); fits[name] = slope * 1000
    xs = np.linspace(0, 270, 50)
    axa.plot(xs, icept + slope * xs, "-", lw=1.4, color=COL[name], alpha=.8, zorder=3)
    axa.plot(n[m], t[m], "o", ms=6.5, color=COL[name], zorder=4,
             label=f"{name}: {slope*1000:.0f} µs per event  ({1000/slope:,.0f} events / s)")
axa.set_xlim(0, 270); axa.set_ylim(0, 56)
axa.set_xlabel("Events per invocation", fontsize=12.5)
axa.set_ylabel("Invocation time [ms]", fontsize=12.5)
axa.legend(frameon=False, fontsize=10.5, loc="upper left")
axa.tick_params(which="both", direction="in", right=True, top=True)
axa.grid(alpha=.13); axa.set_axisbelow(True)
figa.tight_layout()
figa.savefig("/home/snehadri/repos/aie-unsupervised-search/figs/latency_batch_sweep.png", dpi=200, bbox_inches="tight")
figa.savefig("/home/snehadri/repos/aie-unsupervised-search/figs/latency_batch_sweep.pdf", bbox_inches="tight")
print("saved figs/latency_batch_sweep.png")

# (b) PL-only
pl_us = np.array([r[1] for r in ROWS])
axb.barh(y, pl_us, color=PLC, edgecolor=INK, linewidth=0.8, height=0.72, label="PL stage")
for yy, v in zip(y, pl_us):
    axb.text(v + 3, yy, f"{v:.0f}", va="center", fontsize=8.6, color=INK)
# (c) hybrid, stacked
hp = np.array([r[2] for r in ROWS]); ha = np.array([r[3] for r in ROWS])
axc.barh(y, hp, color=AIEC, edgecolor=INK, linewidth=0.8, height=0.72, label="PL stage / PL–AIE streaming")
axc.barh(y, ha, left=hp, color=AIE_DARK, edgecolor=INK, linewidth=0.8, height=0.72, label="AI Engine compute")
for yy, p, a in zip(y, hp, ha):
    # white background so the measured-interval line does not cross the digits
    axc.text(p + a + 3, yy, f"{p+a:.0f}" if a == 0 else f"{p:.0f} + {a:.0f}", va="center",
             fontsize=8.6, color=INK, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

# hybrid panel: the deep-FIFO build's interval (the pipeline with enough buffering
# for its stages to overlap). With the vector integer layer norm the AI Engine
# compute is well under the interval and the PL embedding stage sets the rate.
for ax, meas, letter in ((axb, fits["PL-only"], "(a)"), (axc, fits["AIE-PL hybrid"], "(b)")):
    ax.axvline(meas, color="#c0392b", ls="--", lw=1.6, zorder=5)
    ax.text(meas, len(labs) - 0.35, f"  Measured interval, {meas:.0f} µs", color="#c0392b",
            fontsize=9.5, va="top", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.3)
    ax.set_xlabel("Time per event [µs]", fontsize=11.5)
    ax.set_xlim(0, 245)
    ax.tick_params(axis="x", direction="in", top=True)
    ax.grid(axis="x", alpha=.14); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.text(0.97, 0.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")
axc.legend(frameon=False, fontsize=8.8, loc="lower right", bbox_to_anchor=(1.0, 0.08))

out = "/home/snehadri/repos/aie-unsupervised-search/figs/stage_time_breakdown.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
