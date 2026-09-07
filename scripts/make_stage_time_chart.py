#!/usr/bin/env python
"""Per-stage time for one event, for both deployed designs.

Stage costs come from the routed HLS synthesis reports (csynth.rpt): each
dataflow process's per-event cycle count, converted at the clock the design
actually runs at (80 MHz all-PL, 100 MHz hybrid). Stages in a DATAFLOW region
run concurrently, so these are NOT additive -- the SLOWEST stage sets the
event rate, which is why the measured interval (dashed line) sits at the
tallest bar for the all-PL design.

For the hybrid the tallest PL bar is well short of the measured interval:
the attention now runs on the AI Engine array and does not appear in the PL
report, so the AIE path and its streaming set the rate instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLC, AIEC = "#e8d9a0", "#a9cdea"      # same palette as the dataflow diagrams
INK = "#1a1a1a"

# (label, cycles) in pipeline order; clock applied below
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
PL_MEAS, HYB_MEAS = 205.2, 111.0      # measured per-event interval (batch sweep)

# Measured AIE block intervals (per-block batch sweeps, fit_block_sweeps.py).
# Inserted into the hybrid panel as their own rows so the attention compute
# does not vanish from the picture when it moves off the fabric.
import json, os
_bi = "/home/snehadri/aie_scratch_save_20260810/block_intervals.json"
AIE_ROWS = []
if os.path.isfile(_bi):
    _d = json.load(open(_bi))
    for lab in ("Object attention", "Candidate attention", "Cross attention"):
        if lab in _d:
            AIE_ROWS.append((f"{lab} (AIE)", _d[lab]["slope_us"]))
    # place the AIE rows right after the streaming rows they belong to
    _ins = {"Object attention (AIE)": "Object receive", "Candidate attention (AIE)": "Candidate send/receive",
            "Cross attention (AIE)": "Cross send/receive"}
    for lab, us in AIE_ROWS:
        idx = [i for i, (l, _) in enumerate(HYB) if l == _ins[lab]][0] + 1
        HYB.insert(idx, (lab, us * HYB_CLK / 1e6))     # store as "cycles" so the common path works

plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

for ax, data, clk, meas, col, meas_lab in (
        (axes[0], PL,  PL_CLK,  PL_MEAS,  PLC,  "Measured interval, 205 µs"),
        (axes[1], HYB, HYB_CLK, HYB_MEAS, AIEC, "Measured interval, 111 µs")):
    labs = [d[0] for d in data]
    us = np.array([d[1] / clk * 1e6 for d in data])
    y = np.arange(len(labs))[::-1]
    cols = ["#5b8fc9" if "(AIE)" in l else col for l in labs]      # darker blue = runs on the AIE array
    ax.barh(y, us, color=cols, edgecolor=INK, linewidth=0.8, height=0.72)
    ax.axvline(meas, color="#c0392b", ls="--", lw=1.6, zorder=5)
    ax.text(meas, len(labs) - 0.35, "  " + meas_lab, color="#c0392b",
            fontsize=9.5, va="top", ha="left")
    for yy, v in zip(y, us):
        ax.text(v + 3, yy, f"{v:.0f}", va="center", fontsize=8.8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.5)
    ax.set_xlabel("Time per event [µs]", fontsize=11.5)
    ax.set_xlim(0, 245)
    ax.tick_params(axis="x", direction="in", top=True)
    ax.grid(axis="x", alpha=.14)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/stage_time_breakdown.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
