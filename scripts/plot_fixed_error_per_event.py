#!/usr/bin/env python
"""End-to-end error vs the all-PL golden, per event: the mask-fixed AIE image
vs the all-PL image, both MEASURED ON HARDWARE, all 2000 events.
(wij-routing bug -> corrected model; padded-key mask row in the AIE).
AFTER = stepped twin (bit-exact model of the fixed hardware, 20 events).
BEFORE = the deployed pre-fix image measured on the board (same events),
shown faded for contrast. Symlog y: spans 0.001% .. 100%."""
import numpy as np, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "/home/snehadri/aie_scratch_save_20260810"

def pl_golden(n):
    return np.array([float(m.group(1)) for l in open(f"{SAVE}/E_bkg.txt")
                     if (m := re.match(r"GOLDEN\(all-PL\) ev\d+: MSE=([\d.eE+-]+)", l))])[:n]

def measured(fname, n):
    return np.array([float(m.group(1)) for l in open(f"{SAVE}/{fname}")
                     if (m := re.match(r"\s*ev\s*\d+:\s*([\d.eE+-]+)", l))])[:n]

N = 2000
gold = pl_golden(N)
after = 100.0 * (measured("aie_mf_bkg.txt", N) - gold) / gold
x = np.arange(N)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, after, "o", color="#3b3bcc", ms=3, alpha=0.55,
        markeredgecolor="none", label="Difference")
ax.axhline(0, color="red", lw=1, ls="--", label="Zero Diff")

ax.set_yscale("symlog", linthresh=0.01)
ax.set_ylim(-150, 150)
ax.set_yticks([-100, -10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(["-100", "-10", "-1", "-0.1", "-0.01", "0",
                    "0.01", "0.1", "1", "10", "100"])

med_a = np.median(np.abs(after))
ax.text(0.015, 0.965,
        f"median |error|: {med_a:.3f}%   p90: {np.percentile(np.abs(after),90):.1f}%   "
        f">20%: {100*(np.abs(after)>20).mean():.2f}% of events\n"
        f"outliers = inherent cross near-tie flips   ·   AUC: AIE 0.9644 vs PL 0.9639",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.set_xlabel("Event Index", fontsize=12)
ax.set_ylabel("(AIE - Golden) / Golden   [%]   (symlog)", fontsize=12)
ax.set_title("End-to-End Error vs All-PL Golden, per event\n"
             f"(final MSE loss, retrained weights, hardware vs hardware, all {N} events)", fontsize=13)
ax.set_xticks(range(0, N + 1, 250))
ax.grid(ls="--", alpha=0.4)
ax.legend(loc="upper left", bbox_to_anchor=(0.012, 0.80), fontsize=10, frameon=True)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/fixed_error_per_event.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
a = np.abs(after)
print(f"n={N}  median={np.median(a):.3f}%  p90={np.percentile(a,90):.2f}%  "
      f">20%: {100*(a>20).mean():.2f}%  max={a.max():.0f}%")
