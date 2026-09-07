#!/usr/bin/env python
"""One figure: (a) steady-state invocation time vs batch size for both designs,
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
    "AIE-PL hybrid": [(1,0.81523),(2,0.87427),(4,1.02259),(8,1.59751),(16,2.43396),
                      (32,4.14259),(64,7.83402),(128,14.90631),(256,29.08464)],
}
COL = {"PL-only": PL_C, "AIE-PL hybrid": AIE_C}
NMIN = 8

# ---- (b),(c) per-stage costs from the routed synthesis reports ---------------
PL_CLK, HYB_CLK = 80e6, 100e6
PL = [("Read input", 149), ("Fork", 153), ("Embedding", 4483), ("Pairwise $w_{ij}$", 3028),
      ("Object attention L0", 16269), ("Candidate attention L0", 3499),
      ("Cross attention L0", 13102), ("Object attention L1", 16824),
      ("Candidate attention L1", 3551), ("Cross attention L1", 13102),
      ("Candidate build* + mass", 747), ("Autoencoder + MSE", 792), ("Write DDR", 81)]
HYB = [("Read input", 149), ("Fork", 153), ("Embedding", 5874), ("Pairwise $w_{ij}$", 836),
       ("Remask", 900), ("Object send", 881), ("Object receive", 482),
       ("Candidate build", 1840), ("Candidate send/receive", 250),
       ("Cross send/receive", 1098), ("Candidate build* + mass", 747),
       ("Autoencoder + MSE", 1162), ("Write DDR", 81)]
_bi = "/home/snehadri/aie_scratch_save_20260810/block_intervals.json"
if os.path.isfile(_bi):
    _d = json.load(open(_bi))
    _ins = {"Object attention (AIE)": ("Object receive", "Object attention"),
            "Candidate attention (AIE)": ("Candidate send/receive", "Candidate attention"),
            "Cross attention (AIE)": ("Cross send/receive", "Cross attention")}
    for lab, (after, key) in _ins.items():
        if key in _d:
            idx = [i for i, (l, _) in enumerate(HYB) if l == after][0] + 1
            HYB.insert(idx, (lab, _d[key]["slope_us"] * HYB_CLK / 1e6))

plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(12.6, 10.2))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.55)
axa = fig.add_subplot(gs[0, :]); axb = fig.add_subplot(gs[1, 0]); axc = fig.add_subplot(gs[1, 1])

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
axa.text(0.985, 0.05, "(a)", transform=axa.transAxes, fontsize=12, fontweight="bold", ha="right")

for ax, data, clk, meas, col, letter in (
        (axb, PL,  PL_CLK,  fits["PL-only"],       PLC,  "(b)"),
        (axc, HYB, HYB_CLK, fits["AIE-PL hybrid"], AIEC, "(c)")):
    labs = [d[0] for d in data]
    us = np.array([d[1] / clk * 1e6 for d in data])
    y = np.arange(len(labs))[::-1]
    cols = [AIE_DARK if "(AIE)" in l else col for l in labs]
    ax.barh(y, us, color=cols, edgecolor=INK, linewidth=0.8, height=0.72)
    ax.axvline(meas, color="#c0392b", ls="--", lw=1.6, zorder=5)
    ax.text(meas, len(labs) - 0.35, f"  Measured interval, {meas:.0f} µs", color="#c0392b",
            fontsize=9.5, va="top", ha="left")
    for yy, v in zip(y, us):
        ax.text(v + 3, yy, f"{v:.0f}", va="center", fontsize=8.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.3)
    ax.set_xlabel("Time per event [µs]", fontsize=11.5)
    ax.set_xlim(0, 245)
    ax.tick_params(axis="x", direction="in", top=True)
    ax.grid(axis="x", alpha=.14); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(0.97, 0.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")

out = "/home/snehadri/repos/aie-unsupervised-search/figs/latency_and_stages.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
