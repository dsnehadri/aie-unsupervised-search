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
#   PL-only t2e: softmax pipelined, heads batched, narrow multipliers, fast
#   reshape, fabric linears (65.7 us/event, one event 319 us; 78.125 MHz,
#   two layers, 45% LUT). t2a without the last two levers: 79.8 / 363 us.
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
SWEEP["PL-only, attention blocks optimised"] = _csv(_F + "latency_sweep_pl_t2e.csv", "pl_stream_top")
SWEEP["AIE-PL hybrid, vector AIE kernels"] = _csv(_F + "latency_sweep_hybrid_v3.csv", "aie_stream_top")
NMIN = 8

# ---- (b),(c) per-stage costs, SAME rows in both panels ------------------------
# PL-only: routed cycle counts at 80 MHz. Hybrid: PL parts at 100 MHz from the
# routed report; AIE compute = measured per-block interval (blocks3 sweep, L0
# blocks; L1 uses the same kernels and is taken equal). Hybrid attention rows are
# stacked: light = PL streaming / building on the fabric, dark = AI Engine compute.
# The fabric kernel clock is 78.125 MHz, not the 80 MHz we ask v++ for: the
# clock wizard divides its 625 MHz primitive by an integer, so 80 -> 625/8.
# The hybrid's 100 MHz is exact, it uses the platform's clk_pl_0 with no divider.
PL_CLK, HYB_CLK = 78.125e6, 100e6
# PL-only stage cycles are now the REAL per-event costs of the deployed build
# (integer layer norm + OBJ_DATAFLOW), read from the full-design csynth report
# as each stage's event-loop iteration latency. They used to be the float
# layer-norm counts scaled by 128.4/205.2, which was only an estimate.
# Check: they sum to 41040 cycles = 525.3 us against a measured one-event
# latency of 521 us, 0.8% apart, as they should since a single event cannot
# overlap any stages.
# PROVENANCE: these per-stage costs are from the OBJ_DATAFLOW variant, which
# measures 124.1 us/event. Panel (a)'s interval line is the simpler build we
# quote as the baseline, 128.4 us. The two differ by 3%, which is the whole
# measured benefit of the extra pipelining, and it does not change the picture:
# four stages sit at 96-99 us against an interval of either 124 or 128 us. Swap
# SWEEP["PL-only"] for the 124.1 series if the baseline is ever restated.
# FOUR CO-BOTTLENECKS: object L0 7724, cross L0 7588, cross L1 7588, object L1
# 7472, a spread of 3.4%. That is why splitting only the object block bought 3%.
PL_LNFIX = 1.0
_bi = "/home/snehadri/aie_scratch_save_20260810/block_intervals.json"
_d = json.load(open(_bi)) if os.path.isfile(_bi) else {}
AIE = {k: _d[k]["slope_us"] for k in ("Object attention", "Candidate attention", "Cross attention") if k in _d}
cyc = lambda c, clk: c / clk * 1e6
cycpl = lambda c: c / PL_CLK * 1e6 * PL_LNFIX
# rows: (label, PL-only us, hybrid PL-side us, hybrid AIE us)
ROWS = [
 ("Read input",                cycpl(3),     cyc(149, HYB_CLK),                 0),
 ("Fork",                      cycpl(161),   cyc(153, HYB_CLK),                 0),
 ("Embedding",                 cycpl(2186),  cyc(5874, HYB_CLK),                0),
 ("Pairwise $w_{ij}$",         cycpl(2855),  cyc(836, HYB_CLK),                 0),
 ("Object attention L0",       cycpl(7724),  cyc(614+241+450, HYB_CLK),         AIE.get("Object attention", 0)),
 ("Build candidates + candidate attention L0", cycpl(2115), cyc(920+64+61, HYB_CLK),      AIE.get("Candidate attention", 0)),
 ("Cross attention L0",        cycpl(7588),  cyc(308+241, HYB_CLK),             AIE.get("Cross attention", 0)),
 ("Object attention L1",       cycpl(7472),  cyc(267+241+450, HYB_CLK),         AIE.get("Object attention", 0)),
 ("Build candidates + candidate attention L1", cycpl(2169), cyc(920+64+61, HYB_CLK),      AIE.get("Candidate attention", 0)),
 ("Cross attention L1",        cycpl(7588),  cyc(308+241, HYB_CLK),             AIE.get("Cross attention", 0)),
 ("Candidate build* + mass",   cycpl(759),   cyc(747, HYB_CLK),                 0),
 ("Autoencoder + MSE",         cycpl(416),   cyc(1162, HYB_CLK),                0),
 ("Write DDR",                 cycpl(4),     cyc(81, HYB_CLK),                  0),
]
labs = [r[0] for r in ROWS]
y = np.arange(len(labs))[::-1]

plt.rcParams.update({"font.size": 11})
figa, axa = plt.subplots(figsize=(7.2, 5.0))
fig, (axb, axc, axd) = plt.subplots(1, 3, figsize=(18.0, 5.6))
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
    # white background so the measured-interval line does not cross the digits
    axc.text(p + 3, yy + hh/2, f"{p:.0f}", va="center", fontsize=8.2, color=INK, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    if a:
        axc.text(a + 3, yy - hh/2, f"{a:.0f}", va="center", fontsize=8.2, color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# (d) the same hybrid with the embedding stage moved onto the array. Every bar
# is measured except the embedding kernel's AI Engine cost, which comes from an
# aiesimulator profile of the kernel (16.8 us/event) and is drawn hatched. The
# projected interval is then the longest stage, the object attention block.
EMBED_AIE_SIM = 16.8   # simulated kernel; the built design measures 17.3 us/event
dp = hp.copy(); da = ha.copy()
_emb = labs.index("Embedding")
dp[_emb] = cyc(560, HYB_CLK)          # streaming 12x5 in and 12x16 back, as for the attention blocks
da[_emb] = EMBED_AIE_SIM
axd.barh(y + hh/2, dp, color=AIEC, edgecolor=INK, linewidth=0.8, height=hh)
axd.barh(y - hh/2, da, color=AIE_DARK, edgecolor=INK, linewidth=0.8, height=hh)
axd.barh(y[_emb] - hh/2, da[_emb], color=AIE_DARK, edgecolor=INK, linewidth=0.8,
         height=hh, hatch="////", label="AI Engine compute (embedding from simulation)")
for yy, p_, a_ in zip(y, dp, da):
    axd.text(p_ + 3, yy + hh/2, f"{p_:.0f}", va="center", fontsize=8.2, color=INK, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    if a_:
        axd.text(a_ + 3, yy - hh/2, f"{a_:.0f}", va="center", fontsize=8.2, color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
proj = 17.3   # measured, not projected
axd.axvline(proj, color="#c0392b", ls="--", lw=1.6, zorder=5)
axd.text(proj, len(labs) - 0.35, f"  Measured interval, {proj:.0f} µs", color="#c0392b",
         fontsize=9.5, va="top", ha="left")
axd.legend(frameon=False, fontsize=8.8, loc="lower right", bbox_to_anchor=(1.0, 0.08))

# hybrid panel: the deep-FIFO build's interval (the pipeline with enough buffering
# for its stages to overlap). With the vector integer layer norm the AI Engine
# compute is well under the interval and the PL embedding stage sets the rate.
for ax, meas, letter in ((axb, fits["PL-only"], "(a)"), (axc, fits["AIE-PL hybrid"], "(b)"), (axd, None, "(c)")):
    if meas is not None:
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
