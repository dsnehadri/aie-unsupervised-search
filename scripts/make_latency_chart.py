#!/usr/bin/env python
"""Latency vs throughput for both deployed designs, from a batch-size sweep.

Timing one invocation over a range of batch sizes separates the two numbers
that a single throughput figure conflates:

    t(N) = L + N / T

  slope 1/T  -- the steady-state per-event interval (reciprocal of throughput)
  intercept L -- the fixed cost: pipeline fill and drain plus host launch

Measured on the board with src/host_score_dump.cpp's sibling,
src/host_latency_sweep.cpp (min of 30 iterations per point).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PL_C, AIE_C = "#d62728", "#1f77b4"

SWEEP = {
    "PL-only": [(1,0.90110),(2,1.10842),(4,1.51807),(8,2.33875),(16,3.98148),
                (32,7.26646),(64,13.83435),(128,26.96866),(256,53.24013)],
    "AIE-PL hybrid": [(1,0.81523),(2,0.87427),(4,1.02259),(8,1.59751),(16,2.43396),
                      (32,4.14259),(64,7.83402),(128,14.90631),(256,29.08464)],
}
COL = {"PL-only": PL_C, "AIE-PL hybrid": AIE_C}

plt.rcParams.update({"font.size": 12})
# Steady state only: batches large enough that the pipeline is full (N >= 8).
# The slope of invocation time vs batch size is the per-event interval in
# continuous operation; nothing about single-event or fill/drain cost is shown.
NMIN = 8
fig, ax = plt.subplots(figsize=(7.2, 5.0))
fits = {}
for name, pts in SWEEP.items():
    n = np.array([a for a, _ in pts], float); t = np.array([b for _, b in pts], float)
    m = n >= NMIN
    slope, icept = np.polyfit(n[m], t[m], 1)
    fits[name] = (slope, icept)
    xs = np.linspace(0, 270, 50)
    ax.plot(xs, icept + slope * xs, "-", lw=1.4, color=COL[name], alpha=.8, zorder=3)
    ax.plot(n[m], t[m], "o", ms=6.5, color=COL[name], zorder=4,
            label=f"{name}: {slope*1000:.0f} µs per event  ({1000/slope:,.0f} events / s)")
ax.set_xlim(0, 270); ax.set_ylim(0, 56)
ax.set_xlabel("Events per invocation", fontsize=13)
ax.set_ylabel("Invocation time [ms]", fontsize=13)
ax.legend(frameon=False, fontsize=10.5, loc="upper left")
ax.tick_params(which="both", direction="in", right=True, top=True)
ax.grid(alpha=.13); ax.set_axisbelow(True)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/latency_batch_sweep.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
for n, (s, i) in fits.items():
    print(f"  {n:15s} slope {s*1000:6.1f} us/ev  intercept {i*1000:6.0f} us  -> {1000/s:.0f} ev/s")
