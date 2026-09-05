#!/usr/bin/env python
"""Passwd-ABC algorithm dataflow as implemented in the all-PL pipeline.

Same visual language and print sizing as make_vck190_schematic.py:
authored at physical \\textwidth (6.5 in), DejaVu 10/11 pt (optical parity
with 12/13 pt Computer Modern), plain boxes, black stream arrows.

Serpentine flow: top row left-to-right (embed -> ABC layers), bottom row
right-to-left (final candidate build -> autoencoder -> losses -> out).
Dashed outline = the ABC layer, instantiated twice (layer 0 also adds the
pairwise w_ij score bias). Dimension labels are events' actual sizes:
12 objects x 5 features in, 16-wide embeddings, 3 candidates.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

TEXTWIDTH = 6.5
H = 2.95

STAGE  = "#faf3d9"   # PL pipeline stage
ATTN   = "#cfe3f5"   # attention stage
BLACK  = "#1a1a1a"
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

def arrow(p0, p1, lw=1.2, z=4, head=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=head,
                                 color=BLACK, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))

def elbow(pts, lw=1.2, z=3, head=6):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLACK, lw=lw, zorder=z)
    arrow(pts[-2], pts[-1], lw=lw, z=z, head=head)

# ---------------- top row: input -> embed -> ABC layers --------------------
box(0.05, 2.02, 0.62, 2.78)                              # external memory
txt(0.335, 2.40, "Events\n(DDR4)")
arrow((0.62, 2.40), (1.00, 2.40))
txt(0.81, 2.52, "12×5", size=7.5, bbox=True)

box(1.00, 2.02, 1.85, 2.78, fc=STAGE)
txt(1.425, 2.40, "Per-object\nembedding\n3-layer MLP")
arrow((1.85, 2.40), (2.28, 2.40))
txt(2.065, 2.52, "12×16", size=7.5, bbox=True)

# ABC layer container (instantiated twice)
box(2.20, 1.88, 6.45, 2.88, fc="none", lw=1.0, ls=(0, (4, 3)))
txt(2.30, 2.79, "ABC layer  × 2", FST, "bold", ha="left")
txt(6.35, 2.79, "wᵢⱼ bias: layer 0 only", FS, ha="right")

blue = [(2.28, 3.14, "Object\nattention"),
        (3.24, 4.10, "Build\ncandidates"),
        (4.50, 5.36, "Candidate\nattention"),
        (5.46, 6.37, "Cross\nattention")]
for x0, x1, lab in blue:
    fc = STAGE if "Build" in lab else ATTN
    box(x0, 2.02, x1, 2.66, fc=fc, z=3)
    txt((x0 + x1) / 2, 2.34, lab, z=6)
arrow((3.14, 2.34), (3.24, 2.34))
arrow((4.10, 2.34), (4.50, 2.34))
txt(4.30, 2.46, "3×16", size=7.5, bbox=True)
arrow((5.36, 2.34), (5.46, 2.34))

# object embeddings also feed cross attention (queries)
elbow([(2.75, 2.02), (2.75, 1.96), (5.60, 1.96), (5.60, 2.02)])
txt(4.30, 1.945, "12×16", size=7.5, bbox=True)

# ---------------- pairwise w_ij branch -------------------------------------
box(0.60, 1.50, 2.10, 1.82, fc=STAGE)
txt(1.35, 1.66, "Pairwise wᵢⱼ MLP")
elbow([(0.335, 2.02), (0.335, 1.66), (0.60, 1.66)])
elbow([(2.10, 1.66), (2.45, 1.66), (2.45, 2.02)])

# ---------------- down to bottom row ---------------------------------------
elbow([(6.10, 2.02), (6.10, 1.36)])
txt(6.10, 1.70, "12×16", size=7.5, bbox=True)

# ---------------- bottom row (right to left) -------------------------------
box(5.10, 0.60, 6.45, 1.36, fc=STAGE)
txt(5.775, 0.98, "Build candidates*\n+ invariant mass")
arrow((5.10, 0.98), (4.78, 0.98))
txt(4.94, 1.12, "2×14", size=7.5, bbox=True)

box(3.48, 0.60, 4.78, 1.36, fc=STAGE)
txt(4.13, 0.98, "Autoencoder")
arrow((3.48, 0.98), (3.08, 0.98))
txt(3.28, 1.12, "2×14", size=7.5, bbox=True)

box(1.75, 0.60, 3.08, 1.36, fc=STAGE)
txt(2.415, 0.98, "Reconstruction\nMSE loss")
arrow((1.75, 0.98), (0.85, 0.98))

box(0.05, 0.60, 0.85, 1.36)                              # external memory
txt(0.45, 0.98, "Scores\n(DDR4)")

# ---------------- legend ---------------------------------------------------
box(0.05, 0.02, 6.45, 0.46, fc="white", z=6)
ax.add_patch(Rectangle((0.15, 0.18), 0.28, 0.12, facecolor=STAGE,
                       edgecolor=BLACK, lw=0.8, zorder=7))
txt(0.53, 0.24, "Pipeline stage", ha="left", z=8)
ax.add_patch(Rectangle((1.85, 0.18), 0.28, 0.12, facecolor=ATTN,
                       edgecolor=BLACK, lw=0.8, zorder=7))
txt(2.23, 0.24, "Attention stage", ha="left", z=8)
ax.add_patch(Rectangle((3.60, 0.18), 0.28, 0.12, facecolor="white",
                       edgecolor=BLACK, lw=0.8, zorder=7))
txt(3.98, 0.24, "External memory", ha="left", z=8)
ax.add_patch(FancyArrowPatch((5.35, 0.24), (5.63, 0.24), arrowstyle="->",
                             mutation_scale=6, color=BLACK, lw=1.2, zorder=7))
txt(5.73, 0.24, "Dataflow", ha="left", z=8)

out = "/home/snehadri/repos/aie-unsupervised-search/figs/pl_algorithm_flow.png"
fig.savefig(out, dpi=600, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
