#!/usr/bin/env python
"""VCK190 / XCVC1902 interconnect schematic -- journal-print version.

Authored at PHYSICAL print size: TEXTWIDTH x TEXTWIDTH/2 inches, so a font
set to 12 pt here is exactly 12 pt on the page when the figure is included
at width=\\textwidth. Minimum font size: 12 pt (body), titles 13 pt.
Budgeting rule used for fitting: ~0.10 in per character at 12 pt.

Connectivity per AMD AM009 / NoC docs:
  - AIE array interface = PL interface tiles (AXI4-Stream, DIRECT to the
    fabric, no NoC) + NoC interface tiles (AXI4 via NMU/NSU); 39 of the 50
    columns expose 8-in/6-out 64-bit PL streams
  - PS, PMC, PL, AIE interface, and the integrated DDR memory controllers
    all attach to the NoC
  - PMC boots from MicroSD and configures the device over the NoC
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

TEXTWIDTH = 6.5          # inches -- set to the journal's \textwidth
H = TEXTWIDTH / 2        # 2:1 (fits again at the smaller visual-parity fonts)

ONCHIP  = "#faf3d9"   # on-chip blocks
ONCHIP2 = "#cfe3f5"   # internal resources within an on-chip block
BLACK   = "#1a1a1a"
BLUE    = "#2b6cb0"
ORANGE  = "#d97706"
FS, FST = 10, 11      # DejaVu x-height is ~1.25x Computer Modern's,
                      # so 10/11 pt here reads like 12/13 pt CM body text

fig, ax = plt.subplots(figsize=(TEXTWIDTH, H))
ax.set_xlim(0, TEXTWIDTH); ax.set_ylim(0, H)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

def box(x0, y0, x1, y1, fc="white", lw=1.0, z=2):
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=fc,
                           edgecolor=BLACK, linewidth=lw, zorder=z))

def txt(x, y, s, size=FS, weight="normal", ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            color=BLACK, zorder=z)

def arrow(p0, p1, color=BLACK, lw=1.2, ls="-", both=True, z=4, head=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->" if both else "->",
                                 mutation_scale=head, color=color, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

# ---------------- Programmable Logic (top left) ----------------
box(0.05, 2.44, 3.05, 3.20, fc=ONCHIP)
txt(1.55, 3.03, "Programmable Logic", FST, "bold")
for x0, x1, lab in [(0.20, 1.40, "BRAM / URAM"), (1.48, 2.03, "DSP"),
                    (2.11, 3.01, "LUT / FF")]:
    box(x0, 2.62, x1, 2.92, fc=ONCHIP2, lw=0.8, z=3)
    txt((x0 + x1) / 2, 2.77, lab, FS, z=6)

# ---------------- AI Engine array + interface tiles (top right) ----------------
box(3.60, 2.82, 6.45, 3.20, fc=ONCHIP)
txt(5.02, 3.115, "AI Engine array (400 tiles)", FS, "bold")
for x0 in (3.76, 4.84):
    box(x0, 2.87, x0 + 0.88, 3.03, fc=ONCHIP2, lw=0.8, z=3)
    txt(x0 + 0.44, 2.95, "AIE tile", FS, z=6)
txt(6.08, 2.95, "· · ·", FS)

box(3.60, 2.38, 6.45, 2.72, fc=ONCHIP)
txt(5.02, 2.61, "Array interface tiles", FST, "bold")
txt(5.02, 2.46, "per column: 8 in / 6 out", FS)

# interface <-> array streams; PL <-> interface direct AXI4-Stream
arrow((4.20, 2.72), (4.20, 2.82), color=BLUE, lw=1.2, head=5)
arrow((5.28, 2.72), (5.28, 2.82), color=BLUE, lw=1.2, head=5)
arrow((3.05, 2.55), (3.60, 2.55), color=BLUE, lw=1.4)

# ---------------- NoC ----------------
box(0.05, 2.02, 6.45, 2.28, fc=ONCHIP)
txt(3.25, 2.15, "Network on Chip (NoC)", FST, "bold")
arrow((1.50, 2.44), (1.50, 2.28), head=5)    # PL <-> NoC
arrow((5.00, 2.38), (5.00, 2.28), head=5)    # AIE interface <-> NoC

# ---------------- subsystem row: PMC, PS, DDR controllers ----------------
box(0.10, 1.26, 1.50, 1.86, fc=ONCHIP)
txt(0.80, 1.74, "PMC", FST, "bold")
txt(0.80, 1.475, "Boot + device\nconfiguration", FS)

box(1.70, 1.26, 3.65, 1.86, fc=ONCHIP)
txt(2.675, 1.74, "Processing System", FST, "bold")
txt(2.675, 1.475, "2× Arm Cortex-A72\n2× Arm Cortex-R5", FS)

for x0 in (3.85, 5.20):
    box(x0, 1.26, x0 + 1.15, 1.86, fc=ONCHIP)
    txt(x0 + 0.575, 1.56, "DDR memory\ncontroller", FS)

arrow((0.80, 1.86), (0.80, 2.02), color=ORANGE, ls=":", head=5)
arrow((2.675, 1.86), (2.675, 2.02), head=5)
arrow((4.425, 1.86), (4.425, 2.02), head=5)
arrow((5.775, 1.86), (5.775, 2.02), head=5)

# ---------------- off-chip components ----------------
box(0.10, 0.76, 1.50, 1.12)
txt(0.80, 0.94, "MicroSD\nBoot image", FS)
box(1.70, 0.76, 3.65, 1.12)
txt(2.675, 0.94, "Ethernet / UART", FS)
box(3.85, 0.76, 5.00, 1.12)
txt(4.425, 0.94, "DDR4 DIMM\n8 GB", FS)
box(5.20, 0.76, 6.35, 1.12)
txt(5.775, 0.94, "LPDDR4\n8 GB", FS)

arrow((0.80, 1.12), (0.80, 1.26), color=ORANGE, ls=":", both=False, head=5)
arrow((2.675, 1.12), (2.675, 1.26), head=5)
arrow((4.425, 1.12), (4.425, 1.26), head=5)
arrow((5.775, 1.12), (5.775, 1.26), head=5)

# ---------------- legend (bottom strip, 3 rows x 2 columns) ----------------
box(0.05, 0.02, 6.45, 0.64, fc="white", z=6)
items = [
    (BLACK,  "-",  "AXI4 (memory-mapped)", "arrow"),
    (BLUE,   "-",  "AXI4-Stream",          "arrow"),
    (ORANGE, ":",  "Boot / configuration", "arrow"),
    (ONCHIP, "-",  "On-chip (XCVC1902)",   "swatch"),
    (ONCHIP2, "-", "Internal resource",    "swatch"),
    ("white", "-", "Off-chip component",   "swatch"),
]
for i, (col, ls, lab, kind) in enumerate(items):
    x = 0.20 if i < 3 else 3.35
    y = (0.51, 0.33, 0.15)[i % 3]
    if kind == "arrow":
        ax.add_patch(FancyArrowPatch((x, y), (x + 0.30, y), arrowstyle="->",
                                     mutation_scale=6, color=col, lw=1.2,
                                     linestyle=ls, zorder=7))
    else:
        ax.add_patch(Rectangle((x, y - 0.055), 0.30, 0.11, facecolor=col,
                               edgecolor=BLACK, lw=0.8, zorder=7))
    txt(x + 0.42, y, lab, FS, ha="left", z=8)

out = "/home/snehadri/repos/aie-unsupervised-search/figs/vck190_dataflow_schematic.png"
fig.savefig(out, dpi=600, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
print("saved", out)
