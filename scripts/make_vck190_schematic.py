#!/usr/bin/env python
"""VCK190 dataflow schematic -- utilitarian block-diagram style (plain boxes,
black borders, muted fills, color-coded arrows with a legend).

Content follows the deployed system: MicroSD boots BOOT.BIN (PLM: PL
bitstream + AIE CDO), host app on the A72s moves event buffers through the
NoC/DDR4, PL runs the batched dataflow and talks to the AIE attention
graphs over 64-bit PLIO streams through the interface tiles.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# palette (muted / utilitarian)
ONDIE   = "#f0f0f0"   # on-die panels
CREAM   = "#faf3d9"   # AIE tiles / interface row
NOCBLUE = "#cfe3f5"   # NoC bar
PLIOBLU = "#dce9f7"   # PLIO-endpoint chip emphasis
BLACK   = "#1a1a1a"
BLUE    = "#2b6cb0"   # AXI4-Stream
ORANGE  = "#d97706"   # boot / config
GRAY    = "#808080"   # control

fig, ax = plt.subplots(figsize=(16, 9.6))
ax.set_xlim(0, 160); ax.set_ylim(0, 97)
ax.axis("off")

def box(x0, y0, x1, y1, fc="white", lw=1.2, ls="-", z=2):
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=fc,
                           edgecolor=BLACK, linewidth=lw, linestyle=ls, zorder=z))

def txt(x, y, s, size=9, weight="normal", ha="center", va="center",
        color=BLACK, z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            color=color, zorder=z, fontstyle=style)

def arrow(p0, p1, color=BLACK, lw=1.6, ls="-", both=False, z=4, head=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->" if both else "->",
                                 mutation_scale=head, color=color, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

def polyline(pts, color, lw=1.4, ls="-", z=3, arrow_end=True, head=9):
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="->",
                                 mutation_scale=head, color=color, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

# ---------------- board frame ----------------
box(2, 1.5, 158, 93, fc="white", lw=1.8)
txt(4, 91, "VCK190 evaluation board", 11, "bold", ha="left")
txt(80, 95.2, "VCK190 dataflow — hybrid PL + AI-Engine deployment", 12.5, "bold")

# ---------------- AIE array ----------------
box(40, 70, 118, 89, fc=ONDIE)
txt(79, 86.8, "AI Engine array — 400 tiles @ 1.25 GHz", 10, "bold")
for r, y0 in enumerate([80.6, 75.4]):
    for c in range(8):
        x0 = 44 + c * 9.2
        box(x0, y0, x0 + 7.4, y0 + 4.0, fc=CREAM, lw=0.9, z=3)
        txt(x0 + 3.7, y0 + 2.0, "AIE tile", 6.8, z=6)
txt(79, 72.2, "attention graphs: obj / cand / cross × 2 layers  →  72 tiles, int16", 8.5, style="italic")

# ---------------- AIE interface tiles ----------------
box(40, 61, 118, 67.4, fc=CREAM)
txt(79, 65.4, "AIE interface tiles", 9.5, "bold")
txt(79, 63.0, "PLIO: AXI4-Stream ↔ array  (64-bit @ PL clock — 800 MB/s each, wire-limited)", 7.8)
for x in (50, 70, 90, 110):                      # interface ↔ array
    arrow((x, 67.4), (x, 70), color=BLUE, lw=1.4, both=True, head=8)

# clock-domain boundary
ax.plot([36, 122], [59.3, 59.3], color=BLACK, lw=1.0, ls=(0, (4, 3)), zorder=3)
txt(123, 60.3, "AIE clock (1.25 GHz)", 7.5, ha="left")
txt(123, 58.2, "PL clock (100 MHz)", 7.5, ha="left")

# ---------------- PL ----------------
box(30, 38, 130, 56, fc=ONDIE)
txt(32, 53.8, "Programmable Logic (PL) — batched dataflow, 100 MHz", 10, "bold", ha="left")
chips = [
    (33, 44,   "embed"),
    (46, 61,   "pairwise\nMLP (wij)"),
    (63, 83,   "AIE bridge\nsend / recv"),
    (86, 101,  "cand build\n+ Lorentz"),
    (104, 117, "autoencoder\n+ loss"),
    (120, 128, "m_axi\nDMA"),
]
for i, (x0, x1, lab) in enumerate(chips):
    fc = PLIOBLU if "bridge" in lab else "white"
    box(x0, 41, x1, 49, fc=fc, lw=1.0, z=3)
    txt((x0 + x1) / 2, 45, lab, 7.6, z=6)
    if i < len(chips) - 1:
        arrow((x1, 45), (chips[i + 1][0], 45), lw=1.2, head=7)

# PL bridge ↔ interface tiles (PLIO streams)
arrow((69, 49), (69, 61), color=BLUE, lw=1.8)
arrow((77, 61), (77, 49), color=BLUE, lw=1.8)
txt(67.5, 57.2, "x, wij, mask\n(18 streams)", 7.2, ha="right")
txt(78.5, 57.2, "attention\nout", 7.2, ha="left")

# ---------------- NoC ----------------
box(10, 28, 150, 34, fc=NOCBLUE)
txt(80, 31, "Network on Chip (NoC)", 10, "bold")
arrow((38, 38), (38, 34), lw=1.6, both=True)          # PL <-> NoC (in)
arrow((122, 38), (122, 34), lw=1.6, both=True)        # PL <-> NoC (out)
txt(36.5, 36.2, "events in", 7.4, ha="right")
txt(123.5, 36.2, "losses out", 7.4, ha="left")

# ---------------- bottom row: PS + board components ----------------
box(10, 8, 38, 24, fc=ONDIE)
txt(24, 20.8, "Processing System", 9.5, "bold")
txt(24, 17.4, "2× Cortex-A72 (Linux)", 8.2)
txt(24, 14.2, "2× Cortex-R5", 8.2)
txt(24, 10.9, "host app via XRT (ert_polling)", 7.6, style="italic")

box(44, 8, 70, 24)
txt(57, 18.6, "DDR4 DIMM — 8 GB", 9, "bold")
txt(57, 14.0, "event buffers in / out", 8)

box(74, 8, 96, 24)
txt(85, 18.6, "LPDDR4 — 8 GB", 9, "bold")

box(100, 8, 120, 24)
txt(110, 18.6, "MicroSD (Versal)", 9, "bold")
txt(110, 14.0, "BOOT.BIN + eval data", 8)

box(124, 8, 150, 24)
txt(137, 18.6, "Ethernet / UART-JTAG", 8.6, "bold")
txt(137, 14.0, "SSH + console", 8)

# PS / DDR <-> NoC
arrow((24, 24), (24, 28), lw=1.6, both=True)
arrow((57, 24), (57, 28), lw=1.6, both=True)
arrow((85, 24), (85, 28), lw=1.6, both=True)
txt(60, 26, "8.3 GB/s aggregate (measured)", 7.2, ha="left")

# ---------------- boot / config (orange dashed) ----------------
polyline([(110, 8), (110, 4.5), (28, 4.5), (28, 8)], ORANGE, ls="--")
txt(69, 2.6, "PLM boots BOOT.BIN — PL bitstream + AIE CDO (AIE configured at boot, not by XRT)",
    7.6, color=ORANGE)
polyline([(10, 16), (6, 16), (6, 47), (30, 47)], ORANGE, ls="--")
polyline([(6, 47), (6, 79.5), (40, 79.5)], ORANGE, ls="--")
txt(4.6, 52, "config: bitstream + CDO", 7.2, color=ORANGE, ha="center")
ax.texts[-1].set_rotation(90)

# ---------------- control (gray) ----------------
polyline([(137, 8), (137, 6.3), (32, 6.3), (32, 8)], GRAY, lw=1.1)
txt(139, 6.3, "SSH", 7, color=GRAY, ha="left")

# ---------------- legend ----------------
box(122, 69, 157, 92, fc="white", lw=1.2, z=6)
rows = [
    (BLACK,  "-",  "AXI4 memory-mapped (NoC)", "arrow"),
    (BLUE,   "-",  "AXI4-Stream PLIO (64-bit)", "arrow"),
    (ORANGE, "--", "boot / configuration",      "arrow"),
    (GRAY,   "-",  "control (SSH / UART)",      "arrow"),
    (ONDIE,  "-",  "on-die (XCVC1902)",         "swatch"),
    ("white", "-", "board component",           "swatch"),
]
for i, (col, ls, lab, kind) in enumerate(rows):
    y = 89.2 - i * 3.6
    if kind == "arrow":
        ax.add_patch(FancyArrowPatch((124, y), (130, y), arrowstyle="->",
                                     mutation_scale=8, color=col, lw=1.6,
                                     linestyle=ls, zorder=7))
    else:
        ax.add_patch(Rectangle((124.6, y - 1.1), 4.6, 2.2, facecolor=col,
                               edgecolor=BLACK, lw=0.9, zorder=7))
    txt(131.5, y, lab, 7.8, ha="left", z=8)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/vck190_dataflow_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("saved", out)
