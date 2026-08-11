#!/usr/bin/env python
"""Final-result deviation per event on the NON-collapsed (retrained) model:
AIE hardware (BOOT.BIN.aie_retrained) vs the all-PL hardware (bit-consistent
with the PL golden), same weights, same events. One dot per event, y =
relative difference of the final MSE loss. Companion to
figs/final_result_error.png (the collapsed-weights version, +/-90% chaos)."""
import numpy as np, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "/home/snehadri/aie_scratch_save_20260810"
N_SHOW = 100  # dots shown (stats box quotes all 2000)

def L(f):
    return np.array([float(m.group(1)) for l in open(f) if (m := re.match(r"\s*ev\s*\d+:\s*([\d.eE+-]+)", l))])

aie = L(f"{SAVE}/aie_rt_bkg.txt")
pl  = L(f"{SAVE}/hw2_bkg.txt")
n = min(len(aie), len(pl)); aie, pl = aie[:n], pl[:n]
rel = 100.0 * (aie - pl) / pl
x = np.arange(N_SHOW)

YLIM = 130
shown = rel[:N_SHOW]
inside = np.clip(shown, -YLIM + 5, YLIM - 5)
out_hi = shown > YLIM - 5
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x[~out_hi], shown[~out_hi], "o", color="#3b3bcc", ms=5, alpha=0.75,
        markeredgecolor="k", markeredgewidth=0.3, label="Difference")
ax.plot(x[out_hi], np.full(out_hi.sum(), YLIM - 8), "^", color="#3b3bcc", ms=7,
        markeredgecolor="k", markeredgewidth=0.3)
for xi, v in zip(x[out_hi], shown[out_hi]):
    ax.annotate(f"+{v:.0f}%", xy=(xi, YLIM - 8), xytext=(xi - 6, YLIM - 28),
                fontsize=8, color="dimgray")
ax.axhline(0, color="red", lw=1, ls="--", label="Zero Diff")
ax.set_ylim(-YLIM, YLIM)

ax.text(0.015, 0.965,
        f"median: {np.median(rel):+.1f}%   |diff| p90: {np.percentile(np.abs(rel),90):.0f}%   "
        f"(all {n} events)\nAUC: AIE 0.9614 vs PL 0.9639",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.text(0.5, 0.72,
        "retrained (non-collapsed) model: distribution barely moves (median +2.6%) --\n"
        "the residual per-event tails are near-tie assignment flips that wash out in the AUC",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="dimgray")
ax.set_xlabel("Event Index", fontsize=12)
ax.set_ylabel("(AIE - PL) / PL   [%]", fontsize=12)
ax.set_title("Final Result: AIE Hardware vs All-PL Hardware, per event\n"
             f"(final MSE loss, RETRAINED weights, first {N_SHOW} of {n} events)", fontsize=13)
ax.grid(ls="--", alpha=0.5)
ax.legend(loc="lower right", fontsize=11)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/final_result_error_retrained.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
print(f"n={n} median={np.median(rel):+.2f}%  mean={rel.mean():+.1f}%  "
      f"|rel| p50={np.percentile(np.abs(rel),50):.1f}% p90={np.percentile(np.abs(rel),90):.1f}% "
      f"max={np.abs(rel).max():.0f}%")
