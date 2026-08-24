#!/usr/bin/env python3
"""AIE<->PL link benchmark charts from the pt_link payload sweeps.

Reads the hardware sweep CSVs from the pt_link build dir (v2 = 64-bit PLIO @
100 MHz incl. the no-AIE loopback control; v3 = 128-bit @ 250 MHz single- and
quad-channel when present) and renders:
  figs/aie_pl_link_latency.{png,pdf}    round-trip time vs payload (log-log)
  figs/aie_pl_link_bandwidth.{png,pdf}  effective bandwidth vs payload
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/snehadri/aie_pt_link"
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")

# categorical slots 1-4 (validated adjacent order), text/surface tokens
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

# series -> (csv file, kernel key, color, label)
SERIES = [
    ("sweep_v2_64bit_100mhz.csv", "ptw",  C["blue"],   "AIE round trip, 64-bit @100 MHz"),
    ("sweep_v2_64bit_100mhz.csv", "lb",   C["orange"], "PL loopback control (no AIE)"),
    ("sweep_v3_hw.csv",           "p128", C["aqua"],   "AIE round trip, 128-bit @250 MHz"),
    ("sweep_v3_hw.csv",           "q512", C["yellow"], "AIE 4×128-bit @250 MHz (aggregate)"),
]

def load(fname, key):
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        return []
    pts = []
    for r in csv.reader(open(path)):
        if r and r[0] == key and r[4] != "-1":
            pts.append((int(r[1]), float(r[4])))  # bytes, min_ms
    return sorted(pts)

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=INK2, alpha=0.15, linewidth=0.6)
    ax.set_axisbelow(True)

def bytes_fmt(b):
    for u, d in (("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= d:
            v = b / d
            return f"{v:g} {u}"
    return f"{b} B"

loaded = [(load(f, k), c, lab) for f, k, c, lab in SERIES]
loaded = [(p, c, l) for p, c, l in loaded if p]

for mode in ("latency", "bandwidth"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    for pts, color, label in loaded:
        xs = [b for b, _ in pts]
        if mode == "latency":
            ys = [t for _, t in pts]
        else:
            ys = [b / (t * 1e-3) / 1e6 for b, t in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor=SURF, markeredgewidth=1,
                label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**i for i in range(6, 25, 3)])
    ax.set_xticklabels([bytes_fmt(2**i) for i in range(6, 25, 3)])
    ax.set_xlabel("payload per direction", color=INK, fontsize=10)
    if mode == "latency":
        ax.set_yscale("log")
        ax.set_ylabel("round-trip time (ms, min)", color=INK, fontsize=10)
        ax.set_title("AIE↔PL round-trip time vs payload (VCK190, measured)",
                     color=INK, fontsize=11, loc="left")
        ax.axhline(0.025, color=INK2, linestyle=":", linewidth=1)
        ax.annotate("α ≈ 25 µs launch/fill floor", xy=(2**6.2, 0.0265),
                    color=INK2, fontsize=8.5, va="bottom")
    else:
        ax.set_ylabel("effective bandwidth (MB/s per direction)", color=INK, fontsize=10)
        ax.set_title("AIE↔PL effective bandwidth vs payload (VCK190, measured)",
                     color=INK, fontsize=11, loc="left")
        have_v3 = any("250" in l for _, _, l in loaded)
        ax.axhline(800, color=INK2, linestyle="--", linewidth=1)
        ax.annotate("wire limit, 64-bit × 100 MHz = 800 MB/s", xy=(2**6.2, 815),
                    color=INK2, fontsize=8.5, va="bottom")
        if have_v3:
            ax.axhline(4000, color=INK2, linestyle="--", linewidth=1)
            ax.annotate("wire limit, 128-bit × 250 MHz = 4 GB/s", xy=(2**6.2, 4060),
                        color=INK2, fontsize=8.5, va="bottom")
            ax.set_yscale("log")
            ax.set_ylabel("effective bandwidth (MB/s per direction, log)",
                          color=INK, fontsize=10)
        else:
            ax.set_ylim(0, 900)
    ax.legend(loc="upper left" if mode == "latency" else "lower right",
              frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"aie_pl_link_{mode}.{ext}"),
                    facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/aie_pl_link_{mode}.png/.pdf ({len(loaded)} series)")
