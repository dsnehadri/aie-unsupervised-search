#!/usr/bin/env python
"""End-to-end error vs the all-PL golden, per event, AFTER the fixes -- both
sides MEASURED ON HARDWARE (mask-fixed AIE image vs all-PL image, 2000 events;
first 100 shown).
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

N = 100
gold = pl_golden(N)
after = 100.0 * (measured("aie_mf_bkg.txt", N) - gold) / gold
before = 100.0 * (measured("aie_rt_bkg.txt", N) - gold) / gold
x = np.arange(N)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, before, "o", color="#b0b0b0", ms=5, alpha=0.8,
        markeredgecolor="gray", markeredgewidth=0.3, label="before fixes (board)")
ax.plot(x, after, "o", color="#3b3bcc", ms=6,
        markeredgecolor="k", markeredgewidth=0.3, label="after fixes (board)")
ax.axhline(0, color="red", lw=1, ls="--", label="Zero Diff")

ax.set_yscale("symlog", linthresh=0.01)
ax.set_ylim(-150, 150)
ax.set_yticks([-100, -10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(["-100", "-10", "-1", "-0.1", "-0.01", "0",
                    "0.01", "0.1", "1", "10", "100"])

med_a, med_b = np.median(np.abs(after)), np.median(np.abs(before))
ax.text(0.015, 0.965,
        f"median |error|:  before {med_b:.1f}%   after {med_a:.3f}%\n"
        f"after-fix outliers = inherent cross near-tie flips (1.15% of 2000 events >20%)",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.set_xlabel("Event Index", fontsize=12)
ax.set_ylabel("(AIE - Golden) / Golden   [%]   (symlog)", fontsize=12)
ax.set_title("End-to-End Error vs All-PL Golden, per event\n"
             f"(final MSE loss, retrained weights, hardware vs hardware, first {N} of 2000 events)", fontsize=13)
ax.set_xticks(range(0, N, 10))
ax.grid(ls="--", alpha=0.4)
ax.legend(loc="upper left", bbox_to_anchor=(0.012, 0.80), fontsize=10, frameon=True)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/fixed_error_per_event.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
for i in range(N):
    print(f"ev{i:2d}: before {before[i]:+8.2f}%   after {after[i]:+9.4f}%")
