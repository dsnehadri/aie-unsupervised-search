#!/usr/bin/env python
"""VCK190 / XCVC1902 interconnect schematic -- algorithm-agnostic.

Utilitarian block-diagram style: plain boxes, black borders, muted fills,
color-coded arrows with a legend. Connectivity per AMD AM009 / NoC docs:
  - AIE array interface = PL interface tiles (AXI4-Stream, DIRECT to the
    fabric, no NoC) + NoC interface tiles (AXI4 via NMU/NSU)
  - PS, PMC, PL, AIE interface, and the integrated DDR memory controllers
    all attach to the NoC
  - PMC boots from MicroSD and configures the device (config traffic over
    the NoC)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ONCHIP  = "#f0f0f0"
ONCHIP2 = "#e2e2e2"   # on-chip sub-blocks
BLACK   = "#1a1a1a"
BLUE    = "#2b6cb0"
ORANGE  = "#d97706"

fig, ax = plt.subplots(figsize=(12.5, 7.8))
ax.set_xlim(0, 130); ax.set_ylim(0, 80)
ax.axis("off")

def box(x0, y0, x1, y1, fc="white", lw=1.2, z=2):
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=fc,
                           edgecolor=BLACK, linewidth=lw, zorder=z))

def txt(x, y, s, size=9, weight="normal", ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            color=BLACK, zorder=z)

def arrow(p0, p1, color=BLACK, lw=1.5, ls="-", both=True, z=4, head=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->" if both else "->",
                                 mutation_scale=head, color=color, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

# ---------------- AI Engine array ----------------
box(28, 63.5, 94, 78, fc=ONCHIP)
txt(61, 74.9, "AI Engine array (400 tiles)", 10, "bold")
for x0 in (32, 50, 68):
    box(x0, 65.5, x0 + 14, 71, fc=ONCHIP2, lw=0.9, z=3)
    txt(x0 + 7, 68.2, "AIE tile", 7.8, z=6)
txt(87.5, 68.2, "· · ·", 10)

# ---------------- AIE array interface ----------------
box(28, 54, 94, 60, fc=ONCHIP2)
txt(61, 58.2, "AI Engine array interface tiles", 9.5, "bold")
txt(61, 55.9, "PL interface tiles (AXI4-Stream)   ·   NoC interface tiles (AXI4)", 7.8)
for x in (39, 57, 75):
    arrow((x, 60), (x, 63.5), color=BLUE, lw=1.4)
txt(80.5, 61.75, "per column: 6 \u2191 / 4 \u2193  (32-bit)", 7.2, ha="left")

# ---------------- Programmable Logic ----------------
box(20, 34, 102, 45, fc=ONCHIP)
txt(61, 42.9, "Programmable Logic (FPGA fabric)", 10, "bold")
for x0, x1, lab in [(38, 53, "BRAM / URAM"), (57, 67, "DSP"), (71, 84, "LUT / FF")]:
    box(x0, 35.3, x1, 40.8, fc=ONCHIP2, lw=0.9, z=3)
    txt((x0 + x1) / 2, 38.05, lab, 7.6, z=6)

# PL <-> AIE interface: direct AXI4-Stream through the PL interface tiles
arrow((61, 45), (61, 54), color=BLUE, lw=1.6)
txt(63, 49.5, "39 columns \u00d7 8 \u2191 / 6 \u2193  (64-bit)", 7.2, ha="left")

# AIE interface <-> NoC (routed around the PL block; heads only at the blocks)
ax.plot([94, 110], [57, 57], color=BLACK, lw=1.5, zorder=3)
ax.plot([110, 110], [57, 26], color=BLACK, lw=1.5, zorder=3)
arrow((96, 57), (94, 57), both=False)
arrow((110, 28), (110, 26), both=False)

# ---------------- NoC ----------------
box(10, 20, 120, 26, fc=ONCHIP)
txt(65, 23, "Network on Chip (NoC)", 10, "bold")
arrow((61, 34), (61, 26))            # PL <-> NoC

# ---------------- on-chip row: PMC, PS, DDR controllers ----------------
box(12, 10, 26, 17, fc=ONCHIP)
txt(19, 15.1, "PMC", 9, "bold")
txt(19, 12.3, "Boot + device\nconfiguration", 7.2)

box(30, 10, 56, 17, fc=ONCHIP)
txt(43, 15.1, "Processing System", 9, "bold")
txt(43, 12.3, "2× Arm Cortex-A72\n2× Arm Cortex-R5", 7.2)

for x0 in (62, 84):
    box(x0, 10, x0 + 16, 17, fc=ONCHIP)
    txt(x0 + 8, 13.5, "DDR memory\ncontroller", 7.6)

arrow((19, 17), (19, 20), color=ORANGE, ls=":", lw=1.3, head=6.5)  # PMC config
arrow((43, 17), (43, 20), lw=1.3, head=6.5)
arrow((70, 17), (70, 20), lw=1.3, head=6.5)
arrow((92, 17), (92, 20), lw=1.3, head=6.5)

# ---------------- off-chip components ----------------
box(12, 1.5, 26, 7.5)
txt(19, 4.5, "MicroSD\nBoot image", 7.8)
box(30, 1.5, 48, 7.5)
txt(39, 4.5, "Ethernet / UART", 7.8)
box(62, 1.5, 78, 7.5)
txt(70, 4.5, "DDR4 DIMM\n8 GB", 7.8)
box(84, 1.5, 100, 7.5)
txt(92, 4.5, "LPDDR4\n8 GB", 7.8)

arrow((19, 7.5), (19, 10), color=ORANGE, ls=":", both=False, lw=1.3, head=6.5)
arrow((39, 7.5), (39, 10), lw=1.2, head=6.5)
arrow((70, 7.5), (70, 10), lw=1.2, head=6.5)
arrow((92, 7.5), (92, 10), lw=1.2, head=6.5)

# ---------------- legend ----------------
box(99, 61.5, 127, 78, fc="white", z=6)
rows = [
    (BLACK,  "-",  "AXI4 (memory-mapped)",  "arrow"),
    (BLUE,   "-",  "AXI4-Stream",           "arrow"),
    (ORANGE, ":",  "Boot / configuration",  "arrow"),
    (ONCHIP, "-",  "On-chip (XCVC1902)",    "swatch"),
    ("white", "-", "Off-chip component",    "swatch"),
]
for i, (col, ls, lab, kind) in enumerate(rows):
    y = 75.6 - i * 3.0
    if kind == "arrow":
        ax.add_patch(FancyArrowPatch((101, y), (106.5, y), arrowstyle="->",
                                     mutation_scale=8, color=col, lw=1.5,
                                     linestyle=ls, zorder=7))
    else:
        ax.add_patch(Rectangle((101.4, y - 1.0), 4.4, 2.0, facecolor=col,
                               edgecolor=BLACK, lw=0.9, zorder=7))
    txt(108, y, lab, 7.8, ha="left", z=8)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/vck190_dataflow_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("saved", out)
