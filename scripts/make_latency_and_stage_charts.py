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
    # PL-only with the integer layer norm (LN_MODE=5), 2026-09-09. The float
    # version of the same design ran at 205.2 us/event; the fix is the same one
    # the AI Engine kernels got, so both sides of the comparison now have it.
    "PL-only": [(1,0.52672),(2,0.65845),(4,0.91219),(8,1.42699),(16,2.45448),
                (32,4.51056),(64,8.61873),(128,16.83715),(256,33.27150)],
    # cross-event pipelined hybrid, vector integer layer norm, 4 KB AIE stack, 2026-09-09.
    # Its FIFO depth no longer matters: shallow and deep both give 57.9 us/event
    # (shallow 0.39223..15.16503 ms). With the earlier float layer norm the same
    # two builds gave 111.0 and 57.9 us -- the deeper FIFOs were compensating for
    # a slow AI Engine stage, and once that stage is fast the buffering is idle.
    "AIE-PL hybrid": [(1,0.39255),(2,0.45413),(4,0.56779),(8,0.80058),(16,1.26401),
                      (32,2.19006),(64,4.04590),(128,7.75216),(256,15.16600)],
    # embedding moved onto the array, measured 2026-09-10. The fabric embedding
    # stage set the pipeline's rate, so moving it drops the interval to the
    # object attention block. AUC 0.9822 against 0.9818: no physics lost.
    "AIE-PL hybrid, embedding on array": [(1,0.35504),(2,0.37636),(4,0.41030),(8,0.47965),(16,0.61606),
                      (32,0.89252),(64,1.44643),(128,2.54917),(256,4.76108)],
}
COL = {"PL-only": PL_C, "AIE-PL hybrid": AIE_C, "AIE-PL hybrid, embedding on array": "#2ca02c",
       "PL-only, attention blocks optimised": "#ff7f0e", "AIE-PL hybrid, vector AIE kernels": "#9467bd"}

# The two 2026-09-11/12 latency builds, read from their sweep files (figs/):
#   PL-only t2h: softmax pipelined, heads batched, narrow multipliers, fast
#   reshape, fabric linears, row-pipelined integer layer norm on DSPs, wide
#   streams, 100 MHz (26.3 us/event, one event 145 us; two layers, 52% LUT).
#   t2e (78 MHz, without the last two levers): 65.7 / 319 us; t2a: 79.8 / 363.
#   Hybrid smvec: every AI Engine kernel moves its windows 16 lanes at a time and
#   the object head post uses an 8-lane float softmax (11.3 us/event, one event
#   181 us, 160 us with direct-register launch). Hybrid v3 adds single-pass
#   fabric stages: 7.4 us/event, one event 159 us (137 direct). AUC 0.9825 / 0.9825.
def _csv(path, kernel):
    pts = []
    for l in open(path):
        if l.startswith(kernel + ","):
            q = l.split(","); pts.append((int(q[1]), float(q[4])))   # (N, median ms)
    return pts
_F = "/home/snehadri/repos/aie-unsupervised-search/figs/"
SWEEP["PL-only, attention blocks optimised"] = _csv(_F + "latency_sweep_pl_t2h.csv", "pl_stream_top")
SWEEP["AIE-PL hybrid, vector AIE kernels"] = _csv(_F + "latency_sweep_hybrid_v3.csv", "aie_stream_top")
NMIN = 8

# ---- (b),(c) per-stage costs, SAME rows in both panels ------------------------
# 2026-09-13: both designs restated with the latest measured builds.
#   PL-only = plstream_t2h: softmax pipelined, heads batched, narrow multipliers,
#     fast reshape, fabric linears, row-pipelined integer layer norm on DSPs,
#     wide streams, 100 MHz. Per-stage cost = the event loop's iteration latency
#     of each stage from the full-design csynth report (VITIS_LOOP_<line>_1 rows
#     of pl_stream.h: 754 read, 761 fork, 775 embed, 779 pairwise, 783 obj0,
#     787 obj1, 790 cand, 794 cand2, 798 cross (both layers), 802 lorentz, 806 AE,
#     809+822 write). Sum 12,088 cycles = 121 us against a measured one event of
#     145 us minus the ~20 us launch: 3% apart.
#   Hybrid = aie_hybrid_v3: vector window I/O and float softmax in the AI Engine
#     kernels; fabric kernel with narrow multipliers, fabric LN, pipelined lorentz
#     and single-pass candidate build / remask. PL parts = the per-event iteration
#     latency of each stage loop in aie_stream_top_pipe.cpp (VITIS_LOOP_<line>_1:
#     15 read, 22 fork, 30/34 embed send/recv, 38 pairwise, 44/47 obj send/recv,
#     51 remask, 55 candidate build, 58/61 cand send/recv, 65/68 cross send/recv,
#     72 obj L1 send, 80 lorentz, 84 AE, 87/90 write). AI Engine compute = the
#     slowest kernel of each block from the aiesimulator profile (the block's
#     interval; figs/aie_obj_block_profile.txt), which predicted the board
#     within 6% in every earlier check. Both clocks are exactly 100 MHz.
PL_CLK, HYB_CLK = 100e6, 100e6
AIE = {"Object attention": 6.1, "Candidate attention": 1.6, "Cross attention": 6.0}
EMBED_AIE_SIM = 7.2   # embed_mlp, aiesimulator, 8,998 cycles/event
cyc = lambda c, clk: c / clk * 1e6
cycpl = lambda c: c / PL_CLK * 1e6
# rows: (label, PL-only us, hybrid PL-side us, hybrid AIE us)
# Hybrid PL column: the fabric work of a row is several dataflow processes
# (e.g. object L0 = send 699, recv 243, remask 213 cycles) that run as separate
# pipeline stages, so the bar shows the LONGEST of them -- the one that can set
# the rate -- not their sum (a single event pays the sum: 11.5 us for object L0).
# rows: (label, PL-only us, hybrid PL-side us, hybrid AIE us)
ROWS = [
 ("Read input",                cycpl(152),   cyc(152, HYB_CLK),               0),
 ("Fork",                      cycpl(112),   cyc(161, HYB_CLK),               0),
 ("Embedding",                 cycpl(973),   cyc(max(92, 246), HYB_CLK),      EMBED_AIE_SIM),
 ("Pairwise $w_{ij}$",         cycpl(2640),  cyc(661, HYB_CLK),               0),
 ("Object attention L0",       cycpl(1684),  cyc(max(699, 243, 213), HYB_CLK), AIE["Object attention"]),
 ("Build candidates + candidate attention L0", cycpl(708), cyc(max(348, 68, 63), HYB_CLK), AIE["Candidate attention"]),
 ("Cross attention L0",        cycpl(1880),  cyc(max(300, 243), HYB_CLK),     AIE["Cross attention"]),
 ("Object attention L1",       cycpl(1620),  cyc(max(273, 243, 213), HYB_CLK), AIE["Object attention"]),
 ("Build candidates + candidate attention L1", cycpl(719), cyc(max(348, 68, 63), HYB_CLK), AIE["Candidate attention"]),
 ("Cross attention L1",        cycpl(1880),  cyc(max(300, 243), HYB_CLK),     AIE["Cross attention"]),
 ("Candidate build* + mass",   cycpl(373),   cyc(759, HYB_CLK),               0),
 ("Autoencoder + MSE",         cycpl(281),   cyc(429, HYB_CLK),               0),
 ("Write DDR",                 cycpl(86),    cyc(86, HYB_CLK),                0),
]
labs = [r[0] for r in ROWS]
y = np.arange(len(labs))[::-1]

plt.rcParams.update({"font.size": 11})
figa, axa = plt.subplots(figsize=(7.2, 5.0))
fig, (axb, axc) = plt.subplots(1, 2, figsize=(12.4, 5.6))
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
    axb.text(v + 0.4, yy, f"{v:.1f}", va="center", fontsize=8.6, color=INK)
# (c) hybrid. The fabric part and the AI Engine part of an attention stage
# OVERLAP -- they are separate pipeline stages, so the rate follows the longer
# of the two, not their sum. The measurement says so: before the kernel rewrite
# the AI Engine object block took 54.6 us and the fabric part 13 us, yet the
# interval stayed at 57.9 us, matching the embedding stage alone. Had they been
# serialized the interval could not have been below 68 us. Hence side-by-side
# bars, not a stack.
hp = np.array([r[2] for r in ROWS]); ha = np.array([r[3] for r in ROWS])
hh = 0.36
axc.barh(y + hh/2, hp, color=AIEC, edgecolor=INK, linewidth=0.8, height=hh,
         label="PL stage / PL–AIE streaming")
axc.barh(y - hh/2, ha, color=AIE_DARK, edgecolor=INK, linewidth=0.8, height=hh,
         label="AI Engine compute (kernel interval, aiesimulator)")
for yy, p, a in zip(y, hp, ha):
    # white background so the measured-interval line does not cross the digits
    axc.text(p + 0.4, yy + hh/2, f"{p:.1f}", va="center", fontsize=8.2, color=INK, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    if a:
        axc.text(a + 0.4, yy - hh/2, f"{a:.1f}", va="center", fontsize=8.2, color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# hybrid panel: the deep-FIFO build's interval (the pipeline with enough buffering
# for its stages to overlap). With the vector integer layer norm the AI Engine
# compute is well under the interval and the PL embedding stage sets the rate.
for ax, meas, letter in ((axb, fits["PL-only, attention blocks optimised"], "(a)"), (axc, fits["AIE-PL hybrid, vector AIE kernels"], "(b)")):
    if meas is not None:
        ax.axvline(meas, color="#c0392b", ls="--", lw=1.6, zorder=5)
        ax.text(meas, len(labs) - 0.35, f"  Measured interval, {meas:.1f} µs", color="#c0392b",
                fontsize=9.5, va="top", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.3)
    ax.set_xlabel("Time per event [µs]", fontsize=11.5)
    ax.set_xlim(0, 30)
    ax.tick_params(axis="x", direction="in", top=True)
    ax.grid(axis="x", alpha=.14); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.text(0.97, 0.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")
axc.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.35, -0.12), ncol=1)

out = "/home/snehadri/repos/aie-unsupervised-search/figs/stage_time_breakdown.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
