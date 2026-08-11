#!/usr/bin/env python
"""Final-result deviation per event: board (AIE frac11 image, hardware) vs the
all-PL golden (same original weights, same 11-frac inputs). One dot per event,
y = relative difference of the final MSE loss. Functional-verification style.

Context printed on the plot: with the ORIGINAL (collapsed) weights the jet-
assignment argmax is degenerate, so ANY arithmetic difference flips assignments
and swings per-event losses chaotically -- two pure-PL configs differing only
in fixed-point width disagree by a median 82% on the same metric. This plot is
therefore the amplified end-to-end picture, not the per-op quantization error
(see figs/quantization_error.png for that)."""
import numpy as np, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "/home/snehadri/aie_scratch_save_20260810"

def load(f, pat):
    return np.array([float(m.group(1)) for l in open(f) if (m := re.match(pat, l))])

board  = load(f"{SAVE}/aie_f11_dump100.txt", r"\s*ev\d+: ([\d.eE+-]+)")
golden = load(f"{SAVE}/f11_bkg2000.txt", r"GOLDEN\(all-PL\) ev\d+: MSE=([\d.eE+-]+)")[:len(board)]
rel = 100.0 * (board - golden) / golden
x = np.arange(len(rel))

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, rel, "o", color="#3b3bcc", ms=5, alpha=0.75,
        markeredgecolor="k", markeredgewidth=0.3, label="Difference")
ax.axhline(0, color="red", lw=1, ls="--", label="Zero Diff")

ax.text(0.015, 0.965,
        f"Max Abs Diff: {np.abs(rel).max():.0f}%   median: {np.median(rel):+.0f}%",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.text(0.5, 0.93,
        "collapsed weights: degenerate argmax amplifies any arithmetic difference\n"
        "(two pure-PL fixed-point configs disagree by median 82% on this same metric)",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="dimgray")
ax.set_xlabel("Event Index", fontsize=12)
ax.set_ylabel("(Board - Golden) / Golden   [%]", fontsize=12)
ax.set_title("Final Result: AIE Hardware vs All-PL Golden, per event\n"
             "(final MSE loss, original weights, 100 events)", fontsize=13)
ax.grid(ls="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=11)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/final_result_error.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
print(f"n={len(rel)} median={np.median(rel):+.1f}% mean={rel.mean():+.1f}% "
      f"max abs={np.abs(rel).max():.1f}%")
