#!/usr/bin/env python
"""Before/after the 2026-08-11 fixes (model-repo wij routing bug -> corrected
goldens; padded-key attention leak -> mask row): AIE error vs golden at both
levels. Log scale -- the improvements span 2-3 orders of magnitude.
(a) end-to-end final-loss |relative error| per twin event vs the all-PL golden
(b) per-block worst-event max abs error vs the PyTorch float golden"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEFORE, AFTER = "#eb6834", "#2a78d6"          # validated categorical slots 2, 1
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), width_ratios=[5, 3])
fig.patch.set_facecolor(SURF)

# (a) end-to-end |rel err| %, twin vs all-PL golden
ev = np.arange(5)
before = np.array([5.4, 1.7, 13.6, 12.1, 1.2])
after  = np.array([0.011, 0.04, 0.008, 2.7, 0.009])
w = 0.36
a1.set_facecolor(SURF)
a1.bar(ev - w/2, before, w, color=BEFORE, label="before fixes")
a1.bar(ev + w/2, after,  w, color=AFTER,  label="after fixes")
for x, v in zip(ev - w/2, before):
    a1.text(x, v * 1.15, f"{v:g}", ha="center", fontsize=8.5, color=INK)
for x, v in zip(ev + w/2, after):
    a1.text(x, v * 1.15, f"{v:g}", ha="center", fontsize=8.5, color=INK, weight="bold")
a1.set_yscale("log")
a1.set_ylim(4e-3, 60)
a1.set_xticks(ev); a1.set_xticklabels([f"ev{i}" for i in ev])
a1.set_ylabel("end-to-end |error| vs all-PL golden  [%]", fontsize=11, color=INK)
a1.set_title("(a) final loss, per event — up to 1700× smaller",
             fontsize=11.5, weight="bold", loc="left", color=INK)
a1.annotate("inherent cross\nnear-tie flip", xy=(3 + w/2, 2.7), xytext=(3.35, 12),
            fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
a1.legend(frameon=False, fontsize=9.5, loc="upper left")
a1.grid(alpha=0.15, axis="y")

# (b) per-block worst-event max abs error vs float golden
blocks = ["obj", "cand", "cross"]
bx = np.arange(3)
b_before = np.array([1.04, 0.021, 1.52])
b_after  = np.array([0.024, 0.018, 1.20])
a2.set_facecolor(SURF)
a2.bar(bx - w/2, b_before, w, color=BEFORE)
a2.bar(bx + w/2, b_after,  w, color=AFTER)
for x, v in zip(bx - w/2, b_before):
    a2.text(x, v * 1.15, f"{v:g}", ha="center", fontsize=8.5, color=INK)
for x, v in zip(bx + w/2, b_after):
    a2.text(x, v * 1.15, f"{v:g}", ha="center", fontsize=8.5, color=INK, weight="bold")
a2.set_yscale("log")
a2.set_ylim(6e-3, 6)
a2.set_xticks(bx); a2.set_xticklabels(blocks)
a2.set_ylabel("worst-event max abs error vs float golden", fontsize=11, color=INK)
a2.set_title("(b) attention blocks", fontsize=11.5, weight="bold", loc="left", color=INK)
a2.text(2 + w/2, 0.55, "2/20 events;\nrest at floor", ha="center", fontsize=8, color=INK2)
a2.grid(alpha=0.15, axis="y")

for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)

fig.suptitle("AIE error vs golden, before / after the wij-routing + padded-key-mask fixes",
             fontsize=13, weight="bold", y=1.0, color=INK)
fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/fixes_error_reduction.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURF)
print("saved", out)
