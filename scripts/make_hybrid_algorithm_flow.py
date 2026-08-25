#!/usr/bin/env python
"""Passwd-ABC algorithm dataflow as implemented in the PL + AIE hybrid.

Same layout and print sizing as make_pl_algorithm_flow.py. Difference:
fills show WHERE each stage runs -- tan = PL fabric, blue = AI Engine
array -- and blue arrows mark every PL <-> AIE stream crossing (PLIO).
The six attention graphs (object / candidate / cross, x 2 layers) run on
72 AIE tiles; everything else stays in the PL. Candidate building sits in
the PL between AIE stages, so the flow ping-pongs across the boundary.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

TEXTWIDTH = 6.5
H = 3.10

PLC    = "#faf3d9"   # runs in PL fabric
AIEC   = "#cfe3f5"   # runs on the AI Engine array
BLACK  = "#1a1a1a"
BLUE   = "#2b6cb0"   # PL <-> AIE stream crossing
FS, FST = 10, 11

fig, ax = plt.subplots(figsize=(TEXTWIDTH, H))
ax.set_xlim(0, TEXTWIDTH); ax.set_ylim(0, H)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

def box(x0, y0, x1, y1, fc="white", lw=1.0, ls="-", z=2):
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=fc,
                           edgecolor=BLACK, linewidth=lw, linestyle=ls, zorder=z))

def txt(x, y, s, size=FS, weight="normal", ha="center", va="center", z=5,
        bbox=False):
    kw = dict(fontsize=size, fontweight=weight, ha=ha, va=va, color=BLACK,
              zorder=z)
    if bbox:
        kw["bbox"] = dict(facecolor="white", edgecolor="none", pad=0.6)
    ax.text(x, y, s, **kw)

def arrow(p0, p1, color=BLACK, lw=1.2, z=4, head=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=head,
                                 color=color, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))

def elbow(pts, color=BLACK, lw=1.2, z=3, head=6):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z)
    arrow(pts[-2], pts[-1], color=color, lw=lw, z=z, head=head)

Y0 = 0.15  # everything except the legend sits this much higher than in
           # the all-PL figure (taller legend strip below)

# ---------------- top row: input -> embed -> ABC layers --------------------
box(0.05, 2.02+Y0, 0.62, 2.78+Y0)                        # external memory
txt(0.335, 2.40+Y0, "Events\n(DDR4)")
arrow((0.62, 2.40+Y0), (1.00, 2.40+Y0))
txt(0.81, 2.52+Y0, "12×5", size=7.5, bbox=True)

box(1.00, 2.02+Y0, 1.85, 2.78+Y0, fc=PLC)
txt(1.425, 2.40+Y0, "Per-object\nembedding\n2-layer MLP")
arrow((1.85, 2.40+Y0), (2.28, 2.40+Y0), color=BLUE)
txt(2.065, 2.52+Y0, "12×16", size=7.5, bbox=True)

# ABC layer container (instantiated twice)
box(2.20, 1.88+Y0, 6.45, 2.88+Y0, fc="none", lw=1.0, ls=(0, (4, 3)))
txt(2.30, 2.79+Y0, "ABC layer  × 2", FST, "bold", ha="left")
txt(6.35, 2.79+Y0, "wᵢⱼ bias: layer 0 only", FS, ha="right")

stages = [(2.28, 3.14, "Object\nattention", AIEC),
          (3.24, 4.10, "Build\ncandidates", PLC),
          (4.50, 5.36, "Candidate\nattention", AIEC),
          (5.46, 6.37, "Cross\nattention", AIEC)]
for x0, x1, lab, fc in stages:
    box(x0, 2.02+Y0, x1, 2.66+Y0, fc=fc, z=3)
    txt((x0 + x1) / 2, 2.34+Y0, lab, z=6)
arrow((3.14, 2.34+Y0), (3.24, 2.34+Y0), color=BLUE)
arrow((4.10, 2.34+Y0), (4.50, 2.34+Y0), color=BLUE)
txt(4.30, 2.46+Y0, "3×16", size=7.5, bbox=True)
arrow((5.36, 2.34+Y0), (5.46, 2.34+Y0), color=BLUE)

# object embeddings also feed cross attention (queries)
elbow([(2.75, 2.02+Y0), (2.75, 1.96+Y0), (5.60, 1.96+Y0), (5.60, 2.02+Y0)],
      color=BLUE)
txt(4.30, 1.945+Y0, "12×16", size=7.5, bbox=True)

# ---------------- pairwise w_ij branch -------------------------------------
box(0.60, 1.50+Y0, 2.10, 1.82+Y0, fc=PLC)
txt(1.35, 1.66+Y0, "Pairwise wᵢⱼ MLP")
elbow([(0.335, 2.02+Y0), (0.335, 1.66+Y0), (0.60, 1.66+Y0)])
elbow([(2.10, 1.66+Y0), (2.45, 1.66+Y0), (2.45, 2.02+Y0)], color=BLUE)

# AIE footprint note (below the container, right side)
txt(5.95, 1.78+Y0, "6 attention graphs · 72 AIE tiles", ha="right")

# ---------------- down to bottom row ---------------------------------------
elbow([(6.10, 2.02+Y0), (6.10, 1.36+Y0)], color=BLUE)
txt(6.10, 1.70+Y0, "12×16", size=7.5, bbox=True)

# ---------------- bottom row (right to left) -------------------------------
box(5.10, 0.60+Y0, 6.45, 1.36+Y0, fc=PLC)
txt(5.775, 0.98+Y0, "Build candidates*\n+ invariant mass")
arrow((5.10, 0.98+Y0), (4.78, 0.98+Y0))
txt(4.94, 1.12+Y0, "2×14", size=7.5, bbox=True)

box(3.48, 0.60+Y0, 4.78, 1.36+Y0, fc=PLC)
txt(4.13, 0.98+Y0, "Autoencoder\nshared weights\n× 2 candidates")
arrow((3.48, 0.98+Y0), (3.08, 0.98+Y0))
txt(3.28, 1.12+Y0, "2×14", size=7.5, bbox=True)

box(1.75, 0.60+Y0, 3.08, 1.36+Y0, fc=PLC)
txt(2.415, 0.98+Y0, "MSE losses\nreco + crossed")
arrow((1.75, 0.98+Y0), (0.85, 0.98+Y0))

box(0.05, 0.60+Y0, 0.85, 1.36+Y0)                        # external memory
txt(0.45, 0.98+Y0, "Scores\n(DDR4)")

# ---------------- legend (2 rows) ------------------------------------------
box(0.05, 0.02, 6.45, 0.66, fc="white", z=6)
row1, row2 = 0.48, 0.19
ax.add_patch(Rectangle((0.15, row1-0.06), 0.28, 0.12, facecolor=PLC,
                       edgecolor=BLACK, lw=0.8, zorder=7))
txt(0.53, row1, "Stage in PL fabric", ha="left", z=8)
ax.add_patch(Rectangle((2.35, row1-0.06), 0.28, 0.12, facecolor=AIEC,
                       edgecolor=BLACK, lw=0.8, zorder=7))
txt(2.73, row1, "Stage on AIE array", ha="left", z=8)
ax.add_patch(Rectangle((4.55, row1-0.06), 0.28, 0.12, facecolor="white",
                       edgecolor=BLACK, lw=0.8, zorder=7))
txt(4.93, row1, "External memory", ha="left", z=8)
ax.add_patch(FancyArrowPatch((0.15, row2), (0.43, row2), arrowstyle="->",
                             mutation_scale=6, color=BLACK, lw=1.2, zorder=7))
txt(0.53, row2, "Dataflow in PL", ha="left", z=8)
ax.add_patch(FancyArrowPatch((2.35, row2), (2.63, row2), arrowstyle="->",
                             mutation_scale=6, color=BLUE, lw=1.2, zorder=7))
txt(2.73, row2, "PL ↔ AIE stream (PLIO, 64-bit)", ha="left", z=8)

out = "/home/snehadri/repos/aie-unsupervised-search/figs/hybrid_algorithm_flow.png"
fig.savefig(out, dpi=600, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
