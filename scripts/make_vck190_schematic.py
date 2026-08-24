#!/usr/bin/env python
"""VCK190 / XCVC1902 interconnect schematic -- algorithm-agnostic, 2:1.

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

ONCHIP  = "#faf3d9"   # on-chip blocks
ONCHIP2 = "#cfe3f5"   # internal resources within an on-chip block
BLACK   = "#1a1a1a"
BLUE    = "#2b6cb0"
ORANGE  = "#d97706"

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 200); ax.set_ylim(0, 100)
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

# ---------------- Programmable Logic (top left) ----------------
box(6, 52, 106, 96, fc=ONCHIP)
txt(56, 91.2, "Programmable Logic (FPGA fabric)", 11, "bold")
for x0, x1, lab in [(16, 42, "BRAM / URAM"), (46, 66, "DSP"), (70, 96, "LUT / FF")]:
    box(x0, 60, x1, 74, fc=ONCHIP2, lw=1.0, z=3)
    txt((x0 + x1) / 2, 67, lab, 8.6, z=6)

# ---------------- AI Engine array + interface tiles (top right) ----------------
box(118, 68, 194, 96, fc=ONCHIP)
txt(156, 91.6, "AI Engine array (400 tiles)", 11, "bold")
for x0 in (123, 145, 167):
    box(x0, 72, x0 + 17, 81, fc=ONCHIP2, lw=1.0, z=3)
    txt(x0 + 8.5, 76.5, "AIE tile", 8.2, z=6)
txt(190, 76.5, "·  ·  ·", 9)

box(118, 52, 194, 62, fc=ONCHIP)
txt(156, 59.1, "AI Engine array interface tiles", 9.8, "bold")
txt(156, 55.4, "PL interface tiles (AXI4-Stream)  ·  NoC interface tiles (AXI4)", 8)

# interface <-> array (vertical streams, per column)
for x in (127, 146):
    arrow((x, 62), (x, 68), color=BLUE, lw=1.5)
txt(192, 65, "per column: 6 ↑ / 4 ↓  (32-bit)", 7.6, ha="right")

# PL <-> interface (direct AXI4-Stream through the PL interface tiles)
arrow((106, 57), (118, 57), color=BLUE, lw=1.8)
txt(112, 64, "39 columns\n× 8 → / 6 ←\n(64-bit)", 7.6)

# ---------------- NoC ----------------
box(6, 30, 194, 38, fc=ONCHIP)
txt(100, 34, "Network on Chip (NoC)", 11, "bold")
arrow((50, 52), (50, 38))            # PL <-> NoC
arrow((150, 52), (150, 38))          # AIE interface <-> NoC

# ---------------- on-chip row: PMC, PS, DDR controllers ----------------
box(10, 16, 28, 26, fc=ONCHIP)
txt(19, 23.2, "PMC", 9.2, "bold")
txt(19, 19.4, "Boot + device\nconfiguration", 7.4)

box(34, 16, 64, 26, fc=ONCHIP)
txt(49, 23.2, "Processing System", 9.2, "bold")
txt(49, 19.4, "2× Arm Cortex-A72\n2× Arm Cortex-R5", 7.4)

for x0 in (72, 100):
    box(x0, 16, x0 + 20, 26, fc=ONCHIP)
    txt(x0 + 10, 21, "DDR memory\ncontroller", 8)

arrow((19, 26), (19, 30), color=ORANGE, ls=":", lw=1.3, head=6.5)  # PMC config
arrow((49, 26), (49, 30), lw=1.3, head=6.5)
arrow((82, 26), (82, 30), lw=1.3, head=6.5)
arrow((110, 26), (110, 30), lw=1.3, head=6.5)

# ---------------- off-chip components ----------------
box(10, 2, 28, 10)
txt(19, 6, "MicroSD\nBoot image", 7.8)
box(34, 2, 64, 10)
txt(49, 6, "Ethernet / UART", 8)
box(72, 2, 92, 10)
txt(82, 6, "DDR4 DIMM\n8 GB", 7.8)
box(100, 2, 120, 10)
txt(110, 6, "LPDDR4\n8 GB", 7.8)

arrow((19, 10), (19, 16), color=ORANGE, ls=":", both=False, lw=1.3, head=6.5)
arrow((49, 10), (49, 16), lw=1.2, head=6.5)
arrow((82, 10), (82, 16), lw=1.2, head=6.5)
arrow((110, 10), (110, 16), lw=1.2, head=6.5)

# ---------------- legend ----------------
box(152, 2, 194, 26, fc="white", z=6)
rows = [
    (BLACK,  "-",  "AXI4 (memory-mapped)",  "arrow"),
    (BLUE,   "-",  "AXI4-Stream",           "arrow"),
    (ORANGE, ":",  "Boot / configuration",  "arrow"),
    (ONCHIP, "-",  "On-chip (XCVC1902)",    "swatch"),
    (ONCHIP2, "-", "Internal resource",     "swatch"),
    ("white", "-", "Off-chip component",    "swatch"),
]
for i, (col, ls, lab, kind) in enumerate(rows):
    y = 23.2 - i * 3.7
    if kind == "arrow":
        ax.add_patch(FancyArrowPatch((155, y), (162, y), arrowstyle="->",
                                     mutation_scale=8, color=col, lw=1.5,
                                     linestyle=ls, zorder=7))
    else:
        ax.add_patch(Rectangle((155.4, y - 1.2), 5.6, 2.4, facecolor=col,
                               edgecolor=BLACK, lw=0.9, zorder=7))
    txt(164.5, y, lab, 8, ha="left", z=8)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/vck190_dataflow_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("saved", out)
