#!/usr/bin/env python3
"""Tile-replication scaling plots, driven entirely by the obj16 HW sweep.

All points MEASURED on VCK190 (obj16_emu, n_inst swept 1..16 in one process,
13 AIE tiles per instance). Two figures:
  aie_scaling_throughput  -- aggregate throughput vs tiles: measured (solid) vs
                             ideal-linear (dashed), speedup annotated, PL ref.
  aie_scaling_efficiency  -- per-instance rate + scaling efficiency (% of ideal)
                             vs tiles: shows the shared PL bridge/DMA saturating.
Convention: PL = red, AIE = blue.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid")
except Exception:
    pass
plt.rcParams.update({
    "font.family": "Montserrat", "font.size": 11,
    "axes.titlesize": 12.5, "axes.labelsize": 11.5,
    "figure.dpi": 110, "savefig.dpi": 220,
})

AIE_C, PL_C = "#1f77b4", "#d62728"

# ---- MEASURED obj16 sweep (VCK190, 100 MHz, iters=30, all outputs correct) ----
INST  = [1, 2, 4, 8, 16]
TILES = [13*n for n in INST]                       # 13 tiles / instance
THR   = [4647.7, 8701.3, 15225.9, 24965.7, 36659.3]   # aggregate ev/s
PER   = [4647.7, 4350.7, 3806.5, 3120.7, 2291.2]      # per-instance ev/s
PL_THR = 6334.0                                     # measured PL obj (pl_attn)

RATE   = THR[0] / TILES[0]                          # ideal per-tile rate (from 1x)
IDEAL  = [RATE*t for t in TILES]
SPEEDUP = [t/THR[0] for t in THR]                   # vs 1x
EFF    = [100.0*THR[i]/IDEAL[i] for i in range(len(THR))]  # % of ideal


def fig_throughput(path):
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    # ideal-linear upper bound
    ax.plot([0]+TILES, [0]+IDEAL, "--", color=AIE_C, lw=1.8, alpha=0.5,
            label="ideal (linear scaling)")
    # measured
    ax.plot(TILES, THR, "-o", color=AIE_C, lw=2.2, ms=10, label="AIE (measured)")
    # PL single-instance reference
    ax.axhline(PL_THR, color=PL_C, ls="--", lw=2, label="PL single block (measured)")
    # speedup annotations at each measured point
    for t, y, s in zip(TILES, THR, SPEEDUP):
        ax.annotate(f"{s:.1f}×", (t, y), textcoords="offset points",
                    xytext=(6, -14), fontsize=9.5, weight="bold", color=AIE_C)
    ax.set_xlim(0, 220); ax.set_ylim(0, max(IDEAL)*1.05)
    ax.set_xlabel("AIE tiles used  (13 / instance)")
    ax.set_ylabel("aggregate throughput  [events / s]")
    # secondary x-axis: instance count
    sec = ax.secondary_xaxis("top", functions=(lambda x: x/13.0, lambda x: x*13.0))
    sec.set_xlabel("parallel instances"); sec.set_xticks(INST)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Object attention: throughput scaling with AIE tiles",
                 fontsize=12.5, weight="bold", pad=22)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_efficiency(path):
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    # per-instance throughput (left axis) -- falls as the shared bridge saturates
    l1 = ax.plot(TILES, PER, "-o", color=AIE_C, lw=2.2, ms=10,
                 label="per-instance throughput")
    ax.set_xlabel("AIE tiles used  (13 / instance)")
    ax.set_ylabel("per-instance throughput  [events / s]", color=AIE_C)
    ax.tick_params(axis="y", labelcolor=AIE_C)
    ax.set_ylim(0, PER[0]*1.12); ax.set_xlim(0, 220)
    # scaling efficiency (right axis) -- % of ideal linear
    ax2 = ax.twinx()
    l2 = ax2.plot(TILES, EFF, "-s", color="#555555", lw=2, ms=8,
                  label="scaling efficiency")
    ax2.set_ylabel("scaling efficiency  [% of ideal]", color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.set_ylim(0, 105)
    for t, e in zip(TILES, EFF):
        ax2.annotate(f"{e:.0f}%", (t, e), textcoords="offset points",
                     xytext=(4, 6), fontsize=9, color="#555555")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x/13.0, lambda x: x*13.0))
    sec.set_xlabel("parallel instances"); sec.set_xticks(INST)
    ls = l1 + l2
    ax.legend(ls, [x.get_label() for x in ls], loc="lower left", fontsize=10)
    ax.set_title("Object attention: scaling efficiency vs AIE tiles\n"
                 "(shared PL bridge / DMA saturates as instances grow)",
                 fontsize=12, weight="bold", pad=22)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


if __name__ == "__main__":
    import os
    figs = os.path.join(os.path.dirname(__file__), "..", "figs")
    fig_throughput(os.path.join(figs, "aie_scaling_throughput"))
    fig_efficiency(os.path.join(figs, "aie_scaling_efficiency"))
