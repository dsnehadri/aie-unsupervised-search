#!/usr/bin/env python3
"""Why more AIE tiles don't help: the hybrid is pipeline-bound, not tile-bound.

All throughput numbers MEASURED on VCK190 (host_aie_timed):
- baseline hybrid                     551  ev/s   (78 tiles)
- + pipelined pairwise MLP           1118  ev/s   (78 tiles)
- + cross-event pipelining (N=100)   7549  ev/s   (78 tiles)   <- 13.7x, same tiles
Standalone attention blocks (bridge_solo / cand_emu): obj 4738, cand 14808 ev/s
-> AIE attention is faster than the pipeline can feed it; tiles sit idle.

Outputs (figs/): aie_tiles_not_the_bottleneck
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)
import matplotlib.font_manager as _fm
if any("ontserrat" in (f or "").lower() for f in _fm.findSystemFonts()):
    plt.rcParams["font.family"] = "Montserrat"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "patch.linewidth": 0.0,
})
RED, G1, G2, GREEN, GRAY = "#1f77b4", "#74c476", "#31a354", "#1a8a2e", "#b0b0b0"
NEUTRAL = "#7f7f7f"
TILES_USED, TILES_AVAIL = 78, 400


def fig(path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    # ---- LEFT: throughput ladder — pipelining the SAME 78 tiles ----------
    labels = ["baseline\nhybrid", "+ pipeline\npairwise MLP",
              "+ pipeline\nevents", "2× AIE tiles,\nno pipelining"]
    vals   = [551, 1118, 7549, 551]
    cols   = [RED, G1, GREEN, GRAY]
    tiles  = ["78 tiles", "78 tiles", "78 tiles", "156 tiles"]
    hatch  = [None, None, None, "//"]
    x = np.arange(4)
    bars = axL.bar(x, vals, color=cols, width=0.62)
    for b, h in zip(bars, hatch):
        if h: b.set_hatch(h)
    ymax = max(vals) * 1.22; axL.set_ylim(0, ymax)
    for xi, v, t in zip(x, vals, tiles):
        axL.text(xi, v + ymax*0.015, f"{v:,}", ha="center", va="bottom",
                 fontsize=11, weight="bold")
        axL.text(xi, ymax*0.03, t, ha="center", va="bottom", fontsize=8.5,
                 weight="bold", color="white" if v > ymax*0.15 else NEUTRAL)
    # the "does nothing" bar callout
    axL.text(3, 551 + ymax*0.07, "extra tiles\nsit idle →\nno change",
             ha="center", va="bottom", fontsize=8.5, weight="bold", color=NEUTRAL)
    # the win arc
    axL.annotate("", xy=(2, 7549), xytext=(0, 551),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=2,
                                 connectionstyle="arc3,rad=-0.3"))
    axL.text(1.0, 5600, "13.7× — all on the\nsame 78 tiles",
             ha="center", color=GREEN, weight="bold", fontsize=11)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("throughput [events / s]")
    axL.set_title("Throughput scaled by PIPELINING, not by adding tiles",
                  fontsize=12, weight="bold", pad=10)

    # ---- RIGHT: tile utilization — 81% idle ------------------------------
    used, idle = TILES_USED, TILES_AVAIL - TILES_USED
    axR.bar([0], [used], width=0.5, color=RED, label=f"used ({used})")
    axR.bar([0], [idle], width=0.5, bottom=[used], color="#e8e8e8",
            label=f"idle ({idle})")
    axR.set_xlim(-0.6, 0.6); axR.set_ylim(0, TILES_AVAIL*1.02)
    axR.set_xticks([]); axR.set_ylabel("AIE compute tiles (of 400)")
    axR.text(0, used/2, f"{used}\nused", ha="center", va="center",
             color="white", weight="bold", fontsize=11)
    axR.text(0, used+idle/2, f"{idle} idle\n({100*idle/TILES_AVAIL:.0f}% of array)",
             ha="center", va="center", color=NEUTRAL, weight="bold", fontsize=11)
    axR.set_title("The array is 81% idle", fontsize=12, weight="bold", pad=10)
    axR.text(0, -TILES_AVAIL*0.11,
             "standalone AIE attention:\nobj 4,738 ev/s · cand 14,808 ev/s\n"
             "→ faster than the pipeline can feed it",
             ha="center", va="top", fontsize=8.6, color=NEUTRAL, style="italic")

    fig.suptitle("AIE is tile-rich but pipeline-bound — more tiles wouldn't help",
                 y=1.03, fontsize=13, weight="bold")
    fig.text(0.5, -0.06,
             "Latency of one event is a serial chain (embed→pairwise→obj→…→AE), so more tiles can't shorten it.  "
             "Throughput is set by the slowest pipeline STAGE (data movement / PL), not by AIE compute —\n"
             "which is why overlapping events on the SAME 78 tiles gave 13.7×, while extra tiles would just idle "
             "(and a 2nd full pipeline doesn't fit the PL fabric).",
             ha="center", fontsize=8.3, color=NEUTRAL)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figs"); os.makedirs(out, exist_ok=True)
    fig(os.path.join(out, "aie_tiles_not_the_bottleneck"))
    print("Done.")
