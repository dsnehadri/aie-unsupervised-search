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
#   PL-only t2i: softmax pipelined, heads batched, narrow multipliers, fast
#   reshape, fabric linears, row-pipelined integer layer norm on DSPs, wide
#   streams, pairwise MLP at II=1, 100 MHz (24.6 us/event, one event 127-136 us;
#   two layers, 53% LUT). t2h without the pairwise lever: 26.3 / 145 us.
#   t2e (78 MHz, without the last two levers): 65.7 / 319 us; t2a: 79.8 / 363.
#   Hybrid smvec: every AI Engine kernel moves its windows 16 lanes at a time and
#   the object head post uses an 8-lane float softmax (11.3 us/event, one event
#   181 us, 160 us with direct-register launch). Hybrid v3 adds single-pass
#   fabric stages: 7.4 us/event, one event 159 us (137 direct). The 'chain' image
#   puts the whole ABC stack on the array: 8.4 us/event, one event 134 us (114
#   direct), scores identical to v3. Row streaming between the post kernels then
#   gives 9.5 us/event, one event 108 us (87 direct). Cross-block row streaming,
#   a vector head gather, one wij port instead of four and a 120 MHz fabric give
#   9.6 us/event, one event 98 us (77.5 direct). Splitting the last post stage
#   and streaming the softmax rows then give 6.5 us/event, one event 76 us
#   (55.7 direct). A three-tile embedding pipeline and the cross block's keys as
#   a stream give 5.5 us/event, one event 66 us (52.2 direct), with the fabric
#   at 125 MHz. Scores identical throughout.
#   AUC 0.9825 / 0.9825.
def _csv(path, kernel):
    pts = []
    for l in open(path):
        if l.startswith(kernel + ","):
            q = l.split(","); pts.append((int(q[1]), float(q[4])))   # (N, median ms)
    return pts
_F = "/home/snehadri/repos/aie-unsupervised-search/figs/"
SWEEP["PL-only, attention blocks optimised"] = _csv(_F + "latency_sweep_pl_t2n.csv", "pl_stream_top")
SWEEP["AIE-PL hybrid, vector AIE kernels"] = _csv(_F + "latency_sweep_hybrid_v6f.csv", "aie_stream_top")
NMIN = 8

# ---- (b),(c) per-stage costs, SAME rows in both panels ------------------------
# 2026-09-13: both designs restated with the latest measured builds.
#   PL-only = plstream_t2i: softmax pipelined, heads batched, narrow multipliers,
#     fast reshape, fabric linears, row-pipelined integer layer norm on DSPs,
#     wide streams, pairwise MLP at II=1, 100 MHz (t2h = the same without the
#     pairwise lever; only the pairwise row differs). Per-stage cost = the event loop's iteration latency
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
PL_CLK, HYB_CLK = 170e6, 133e6   # fabric-only t2p at 170 MHz, the hybrid fabric at 133 (v8)
# t2p is t2n's logic relinked at 170 MHz, so every fabric-only stage keeps its
# cycle count and only the clock changes. The hybrid's fabric did change in v8
# (wide reads, early four-vectors, single-pass norm); its cycle counts below are
# read from that build's own csynth report, not scaled.
# MEASURED on the board, 2026-09-16: each block alone on the array with the v5 flags
# (blocks3_v2, preloaded feeders, batch-sweep slope). The simulator gave 3.8 us for
# both object and cross.
AIE = {"Object attention": 4.1, "Candidate attention": 1.0, "Cross attention": 3.8}
PAIR_AIE_SIM = 2.6    # pairwise bias MLP on the array (v6): 12 chains + merge tree, aiesim interval
EMBED_AIE_SIM = 3.2   # embed_mlp as three tiles: first tile 4,047 cycles/event in the all-levers chain profile
cyc = lambda c, clk: c / clk * 1e6
cycpl = lambda c: c / PL_CLK * 1e6
# rows: (label, PL-only us, hybrid PL-side us, hybrid AIE us)
# 2026-09-16: every fabric number below re-read from the csynth reports of the two
# DEPLOYED builds (plstream_t2k; hybrid v5 = aie_hybrid_v5, fabric from aie_hybrid_ae3), per-event iteration latency of each
# stage loop. Both were scheduled against the same 3.2 ns HLS target and run at
# 156.25 / 125 MHz, so the cycle counts compare directly. Hybrid: read 152, fork 161,
# embed send 92, mask send 29, pairwise 653, wij send 397, x recv 243, c recv 63,
# lorentz 623, AE 306 (local weights: the two decoders no longer take turns), write 84.
# Fabric-only t2l (LIN_J_UNROLL=2, two outputs per cycle in the linear layers): read 152,
# fork 112, embed 669, pairwise 428, obj 1327/1263, cand 540/551, cross 1376, lorentz 373,
# AE 352 (t2k 281), write 84 -- sum 55.1 us vs 53.8 us measured with direct registers.
# 2026-09-17: fabric-only t2n (LIN_J_UNROLL=4): embed 525, obj 1111/1047, cand 456/467,
# cross 1124, AE 330, the rest unchanged; 43.9 us measured direct. Hybrid v6f (133 MHz):
# the pairwise MLP runs on the array (its bias send loop is gone), the Lorentz stage
# reads the array's 64-bit beats (390 cycles), AE 306; 42.4 us measured direct.
# Hybrid = the "chain" image (whole ABC stack on the array, 2026-09-14). PL
# column: per-event iteration latency of each fabric stage loop in
# aie_stream_top_chain.cpp (read 152, fork 161, embed send 92, mask send 29,
# pairwise 661, wij send 482, x recv 243, c recv 63, lorentz 759, AE 429,
# write 86 cycles at 100 MHz); where a row holds several dataflow processes the
# bar is the longest one. The attention blocks have no fabric part any more:
# the mask row, remask and candidate build run on tiles (~0.5 us, folded into
# the candidate rows). AI Engine column: slowest kernel of each block from the
# aiesimulator profiles, taken from the block's output timestamps. The measured
# 9.5 us interval is the array's own rate (the simulator of the whole stack gave
# 9.4): the block kernels plus the assemble/post-obj hops and stream fan-outs
# between them.
#   2026-09-14, row streaming inside each block (POST_STREAM): a_proj -> b1 -> b2
#   -> c pass one 16-word row at a time on core streams. That is a LATENCY lever,
#   not a rate one, so these bars hardly move (object block interval 5.9 -> 6.2,
#   cross 6.4 -> 6.2, candidate 1.7 -> 1.6) while the block's own latency falls:
#   object 24.2 -> 14.9 us, cross 19.7 -> 10.7, candidate 6.7 -> 4.3, whole stack
#   111 -> 77 us in the simulator and 134 -> 108 us on the board (87 direct).
ROWS = [
 ("Read input",                cycpl(152),   cyc(98,  HYB_CLK),          0),
 ("Fork",                      cycpl(112),   cyc(22,  HYB_CLK),          0),   # wide reads: 18 beats at II=1 + 4
 ("Embedding",                 cycpl(525),   cyc(max(92, 29), HYB_CLK),  EMBED_AIE_SIM),
 ("Pairwise $w_{ij}$",         cycpl(428),   0,                          PAIR_AIE_SIM),   # t2i: pairwise at II=1 (t2h: 2640)
 ("Object attention L0",       cycpl(1111),  0,                          AIE["Object attention"]),
 ("Build candidates + candidate attention L0", cycpl(456), 0,            AIE["Candidate attention"] + 0.5),
 ("Cross attention L0",        cycpl(1124),  0,                          AIE["Cross attention"]),
 ("Object attention L1",       cycpl(1047),  0,                          AIE["Object attention"]),
 ("Build candidates + candidate attention L1", cycpl(467), 0,            AIE["Candidate attention"] + 0.5),
 ("Cross attention L1",        cycpl(1124),  0,                          AIE["Cross attention"]),
 ("Candidate build* + mass",   cycpl(373),   cyc(331, HYB_CLK),          0),   # P4_EARLY moves the exponentials out
 ("Autoencoder + MSE",         cycpl(330),   cyc(296, HYB_CLK),          0),
 ("Write DDR",                 cycpl(84),    cyc(84, HYB_CLK),           0),
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
         label="AI Engine compute")
for yy, p, a in zip(y, hp, ha):
    if p:
        axc.text(p + 0.4, yy + hh/2, f"{p:.1f}", va="center", fontsize=8.2, color=INK, zorder=6)
    if a:
        axc.text(a + 0.4, yy - hh/2, f"{a:.1f}", va="center", fontsize=8.2, color=INK, zorder=6)

# Per-stage cost only: the measured-interval marker was removed, so each panel
# shows what the stages cost and nothing else.
_xmax = max(pl_us.max(), hp.max(), ha.max()) * 1.18
for ax, letter in ((axb, "(a)"), (axc, "(b)")):
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.3)
    ax.set_xlabel("Time per event [µs]", fontsize=11.5)
    ax.set_xlim(0, _xmax)
    ax.tick_params(axis="x", direction="in", top=True)
    ax.grid(axis="x", alpha=.14); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.text(0.97, 0.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")
axc.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.35, -0.12), ncol=1)

out = "/home/snehadri/repos/aie-unsupervised-search/figs/stage_time_breakdown.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
