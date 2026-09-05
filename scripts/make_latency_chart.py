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
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.4, 5.0))

fits = {}
for name, pts in SWEEP.items():
    n = np.array([a for a, _ in pts], float)
    t = np.array([b for _, b in pts], float)
    m = n >= 8                                   # steady-state region
    A = np.vstack([n[m], np.ones(m.sum())]).T
    (slope, icept), *_ = np.linalg.lstsq(A, t[m], rcond=None)
    fits[name] = (slope, icept)

    axa.plot(n, t, "o", ms=6, color=COL[name], zorder=4,
             label=f"{name}: {slope*1000:.0f} µs/event")
    xs = np.linspace(0, 270, 50)
    axa.plot(xs, icept + slope * xs, "-", lw=1.4, color=COL[name], alpha=.75, zorder=3)
    axa.plot([0], [icept], "s", ms=6, mfc="white", mec=COL[name], mew=1.6, zorder=5)

    axb.plot(n, t * 1000 / n, "o-", ms=5.5, lw=1.6, color=COL[name], label=name)
    axb.axhline(slope * 1000, ls="--", lw=1.2, color=COL[name], alpha=.65)

axa.set_xlim(0, 270); axa.set_ylim(0, 56)
axa.set_xlabel("Events per invocation", fontsize=13)
axa.set_ylabel("Invocation time [ms]", fontsize=13)
axa.legend(frameon=False, fontsize=10.5, loc="upper left")
# the two intercepts differ by only 25 us, so label them once
_i_pl = fits["PL-only"][1] * 1000
_i_hy = fits["AIE-PL hybrid"][1] * 1000
axa.annotate(f"Fixed cost {_i_hy:.0f}–{_i_pl:.0f} µs\n(pipeline fill/drain + launch)",
             xy=(0, fits["PL-only"][1]), xytext=(46, 7.4),
             fontsize=9.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

axb.set_xscale("log", base=2); axb.set_xlim(0.8, 320); axb.set_ylim(0, 950)
axb.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
axb.set_xticklabels([str(v) for v in (1, 2, 4, 8, 16, 32, 64, 128, 256)])
axb.set_xlabel("Events per invocation", fontsize=13)
axb.set_ylabel("Time per event [µs]", fontsize=13)
axb.legend(frameon=False, fontsize=10.5, loc="upper right")
axb.text(1.05, fits["PL-only"][0]*1000 + 28, "Steady state 205 µs",
         fontsize=9.5, color=PL_C)
axb.text(1.05, fits["AIE-PL hybrid"][0]*1000 - 62, "Steady state 111 µs",
         fontsize=9.5, color=AIE_C)

for ax in (axa, axb):
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(alpha=.13); ax.set_axisbelow(True)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/latency_batch_sweep.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
for n, (s, i) in fits.items():
    print(f"  {n:15s} slope {s*1000:6.1f} us/ev  intercept {i*1000:6.0f} us  -> {1000/s:.0f} ev/s")
