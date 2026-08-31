#!/usr/bin/env python3
"""AIE vs PL throughput (events/second) -- all numbers MEASURED on the VCK190
(host CU start->done wall clock, ert_polling).

Figures (figs/):
  throughput_endtoend   end-to-end pipeline throughput, every deployed config
  throughput_per_block  single-instance attention block, PL vs AIE runtime
  throughput_scaling    AIE obj-block tile-replication sweep (13..208 tiles)
                        vs single-instance PL block
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
PL_C, AIE_C = "#d62728", "#1f77b4"
FIGS = "/home/snehadri/repos/aie-unsupervised-search/figs"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIGS}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight")
    print("saved", f"{FIGS}/{name}.png")


# ---------------------------------------------------------------------------
# 1. End-to-end pipeline throughput (full anomaly model, 2000-event runs)
# ---------------------------------------------------------------------------
def fig_endtoend():
    rows = [  # (label, ev/s, color)
        ("all-PL  baseline",                    478,  PL_C),
        ("all-PL  optimized kernels",          1139,  PL_C),
        ("all-PL  batched dataflow",           4869,  PL_C),
        ("AIE hybrid  baseline",                551,  AIE_C),
        ("AIE hybrid  pipelined bridge",       7549,  AIE_C),
        ("AIE hybrid  current (72 tiles)",     8964,  AIE_C),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows], float)
    cols = [r[2] for r in rows]
    ys = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(ys, vals, color=cols, height=0.62)
    for y, v in zip(ys, vals):
        ax.text(v + 90, y, f"{v:,.0f}", va="center", fontsize=10.5, weight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("throughput  [events / s]")
    ax.set_xlim(0, vals.max() * 1.14)
    ax.set_title("End-to-end throughput on VCK190 (measured, full model)",
                 fontsize=12.5, pad=10)
    handles = [mpl.patches.Patch(color=PL_C, label="all-PL (AUC 0.9639)"),
               mpl.patches.Patch(color=AIE_C, label="AIE hybrid (AUC 0.9644)")]
    ax.legend(handles=handles, loc="lower right", fontsize=10)
    save(fig, "throughput_endtoend")


# ---------------------------------------------------------------------------
# 2. Per-block single-instance throughput, PL vs AIE (runtime @100 MHz)
# ---------------------------------------------------------------------------
def fig_blocks_and_scaling():
    """Two panels: single-block PL vs AIE, and AIE tile replication."""
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    # --- (a) single attention block, PL vs AIE ---
    blocks = ["Object\nattention block", "Candidate\nattention block",
              "Cross\nattention block"]
    pl_v = np.array([6334, 25458, 7307], float)
    aie_v = np.array([4738, 14808, 5212], float)
    xs = np.arange(len(blocks))
    w = 0.36
    axb.bar(xs - w/2, pl_v,  width=w, color=PL_C,  label="PL block")
    axb.bar(xs + w/2, aie_v, width=w, color=AIE_C, label="AIE block")
    axb.set_xticks(xs)
    axb.set_xticklabels(blocks, fontsize=11.5)
    axb.set_ylabel("Throughput [events / s]", fontsize=12.5)
    axb.set_ylim(0, pl_v.max() * 1.12)
    axb.legend(fontsize=11, frameon=False)

    # --- (b) AIE tile replication ---
    tiles = np.array([13, 26, 52, 104, 208], float)
    meas = np.array([4648, 8701, 15226, 24966, 36659], float)
    axs.plot(tiles, meas, "o-", color=AIE_C, lw=2, ms=7,
             markeredgecolor="k", markeredgewidth=0.4, label="AIE object block")
    axs.axhline(6334, color=PL_C, lw=2, ls="--",
                label="PL object block")
    axs.set_xticks(tiles)
    axs.set_xlim(0, tiles.max() * 1.05)
    axs.set_xlabel("AI Engine tiles", fontsize=12.5)
    axs.set_ylabel("Throughput [events / s]", fontsize=12.5)
    axs.set_ylim(0, meas.max() * 1.12)
    axs.legend(fontsize=11, loc="upper left", frameon=False)

    save(fig, "throughput_blocks_and_scaling")


if __name__ == "__main__":
    fig_endtoend()
    fig_blocks_and_scaling()
