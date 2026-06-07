#!/usr/bin/env python3
# Render two plots:
#   figs/kernel_total_latency.png  - per-kernel total latency (overview)
#   figs/kernel_stage_breakdown.png - per-kernel sub-stage breakdown
#
# The breakdown groups sub-instances into categories (Linear, Attention,
# Norm/Residual, FFN, Activation, Glue) and uses each kernel's outer-loop
# trip counts to inflate inside-loop instances so the stacked bar reflects
# the wall-clock time spent in each category.

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
DATA = json.loads((REPO / "scripts/bench/breakdown.json").read_text())
OUT_DIR = REPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

# Match the aie_vs_pl_steady style: seaborn paper theme + Montserrat.
sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)
if any("ontserrat" in (f or "").lower() for f in _fm.findSystemFonts()):
    plt.rcParams["font.family"] = "Montserrat"
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "patch.linewidth": 0.0,
})

PL_COLOR  = "#1f77b4"
AIE_COLOR = "#d62728"
NEUTRAL   = "#7f7f7f"


# -- categorization ----------------------------------------------------
CATEGORIES = [
    ("Linear",        "#3a6fb0"),   # blue
    ("Attention",     "#c0392b"),   # red
    ("Norm/Residual", "#2e8b57"),   # sea green
    ("FFN",           "#8e44ad"),   # purple
    ("Activation",    "#e67e22"),   # orange
    ("AE enc/dec",    "#17becf"),   # cyan
    ("Candidate ops", "#8c564b"),   # brown
    ("Reshape/glue",  "#7f8c8d"),   # gray
    ("Loop overhead", "#bdc3c7"),   # light gray
]
CAT_COLOR = dict(CATEGORIES)


def categorize(module: str) -> str:
    m = module.lower()
    if m.startswith("linear"):
        return "Linear"
    if "softmax" in m or "qk_i_qj_i" in m.replace("_", ""):
        return "Attention"
    if "layernorm" in m or "skip_and_norm" in m:
        return "Norm/Residual"
    if "ffn_block" in m:
        return "FFN"
    if "relu" in m:
        return "Activation"
    if m in {"ae_encode", "ae_decode"}:
        return "AE enc/dec"
    if m in {"get_jet_choice", "x_to_p4_hw", "build_candidates_p4", "assemble_ae_input"}:
        return "Candidate ops"
    if m.startswith("dual_autoencoder_pipeline"):
        return "Reshape/glue"
    if m.startswith("candidate_build_top_pipeline"):
        if "vitis_loop" in m:
            return "Candidate ops"
        return "Reshape/glue"
    # default: anything else (Pipeline_CONCAT, Pipeline_VITIS_LOOP_*, reshape, etc.)
    return "Reshape/glue"


# -- per-kernel loop multipliers --------------------------------------
# For each kernel we identify the outer loop and the modules whose
# instance latency should be multiplied by the outer trip count to match
# wall-clock time spent in that module. Determined by reading the .h
# source files.
LOOP_MULTIPLIERS = {
    # kernel: list of (module_name_substring, trip_count)
    "pairwise_mlp": [
        # PAIR_I_PAIR_J iterates 144 times, all modules listed in instances run inside
        ("pairwise_mlp_Pipeline_VITIS_LOOP_19_1_VITIS_LOOP_24_2", 1),  # this is the outer pair index gen, runs once
        ("pairwise_mlp_Pipeline_LIN_J", 144),
        ("layernorm_1_16", 144),
        ("relu_2d_1_16", 144),
        ("linear_1_16_16", 144),
        ("linear_1_1_16", 144),
        ("layernorm_1_16_1", 144),
    ],
    "attn_obj": [
        # HEAD_LOOP trip=4; per-head modules: softmax_and_context, plus small loops 92_4/101_6
        ("softmax_and_context", 4),
        ("attn_block_obj_Pipeline_VITIS_LOOP_92_4", 4),
        ("attn_block_obj_Pipeline_VITIS_LOOP_101_6", 4),
        ("attn_block_obj_Pipeline_QK_I_QJ_I", 4),
    ],
    "attn_cand": [
        ("softmax_and_context", 4),
        ("attn_block_cand_Pipeline_QK_I_QJ_I", 4),
        # cand has trip=3 over heads? actually it's 4 heads too; let me match attn_obj
    ],
    "attn_cross": [
        ("softmax_and_context", 4),
        ("attn_block_cross_Pipeline_QK_I_QJ_I", 4),
    ],
}


def cycles_to_us(cycles, period_ns):
    if cycles is None:
        return 0.0
    return cycles * (period_ns or 5.0) / 1000.0


def aggregate(d):
    """Return dict[category] -> microseconds, scaled to match top latency."""
    period = d["period_ns"] or 5.0
    top_us = cycles_to_us(d["top_cycles_min"], period)
    cat_us = {c: 0.0 for c, _ in CATEGORIES}

    multipliers = LOOP_MULTIPLIERS.get(d["kernel"], [])

    for inst in d["instances"]:
        mod = inst["module"]
        cyc = inst["cycles_min"]
        mult = 1
        for substr, m in multipliers:
            if substr in mod:
                mult = m
                break
        us = cycles_to_us(cyc * mult, period)
        cat = categorize(mod)
        cat_us[cat] += us

    sum_so_far = sum(cat_us.values())
    if sum_so_far > top_us and sum_so_far > 0:
        # instances overlap (dataflow/parallelism); scale down to fit
        scale = top_us / sum_so_far
        for c in cat_us:
            cat_us[c] *= scale
    else:
        cat_us["Loop overhead"] += (top_us - sum_so_far)
    return cat_us, top_us


# -- Plot 1: totals ----------------------------------------------------
def plot_totals():
    kernels = [d["kernel"] for d in DATA]
    us_min = [cycles_to_us(d["top_cycles_min"], d["period_ns"]) for d in DATA]
    us_max = [cycles_to_us(d["top_cycles_max"], d["period_ns"]) for d in DATA]

    order = sorted(range(len(kernels)), key=lambda i: -us_min[i])
    kernels = [kernels[i] for i in order]
    us_min = [us_min[i] for i in order]
    us_max = [us_max[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    y = np.arange(len(kernels))
    ax.barh(y, us_min, color=PL_COLOR, edgecolor="none")
    for i, (lo, hi) in enumerate(zip(us_min, us_max)):
        label = f"{lo:.1f} µs" if lo == hi else f"{lo:.1f}–{hi:.1f} µs"
        ax.text(lo * 1.05, i, label, va="center",
                fontsize=9.5, weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(kernels)
    ax.set_xscale("log")
    ax.set_xlabel("latency per invocation  [μs, log scale]")
    ax.set_ylabel("")
    ax.set_title("PL kernel latency — HLS csynth estimates "
                 "(xcvc1902, target 5 ns)",
                 fontsize=12, weight="bold", pad=10)
    ax.grid(True, axis="x", which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    out = OUT_DIR / "kernel_total_latency.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT_DIR / "kernel_total_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -- Plot 2: stacked breakdown -----------------------------------------
def plot_breakdown():
    # sort kernels by total latency desc for consistent layout
    order = sorted(DATA, key=lambda d: -cycles_to_us(d["top_cycles_min"], d["period_ns"]))
    kernels = [d["kernel"] for d in order]
    totals = [cycles_to_us(d["top_cycles_min"], d["period_ns"]) for d in order]
    aggregates = [aggregate(d)[0] for d in order]

    # ---- absolute µs version ---------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    y = np.arange(len(kernels))

    left = np.zeros(len(kernels))
    for cat, color in CATEGORIES:
        widths = np.array([a[cat] for a in aggregates])
        ax.barh(y, widths, left=left, color=color, edgecolor="white",
                linewidth=0.5, label=cat)
        for i, w in enumerate(widths):
            if w / totals[i] > 0.05 and w > 0.5:
                ax.text(left[i] + w / 2, i, f"{w:.1f}",
                        va="center", ha="center", fontsize=8.5,
                        color="white" if cat != "Loop overhead" else "#333",
                        weight="bold")
        left += widths

    for i, t in enumerate(totals):
        ax.text(t * 1.008, i, f"{t:.1f} μs",
                va="center", ha="left",
                fontsize=10, weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(kernels)
    ax.set_xlabel("wall-clock time per invocation  [μs]")
    ax.set_ylabel("")
    ax.set_title("PL kernel per-stage breakdown — csynth, scaled to event total",
                 fontsize=12, weight="bold", pad=10)
    ax.set_xlim(0, max(totals) * 1.14)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, ncol=3, framealpha=0.95,
              frameon=True, edgecolor="#ccc")
    out = OUT_DIR / "kernel_stage_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT_DIR / "kernel_stage_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---- normalized 0..100% version --------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    left = np.zeros(len(kernels))
    for cat, color in CATEGORIES:
        widths = np.array([a[cat] / totals[i] * 100.0 for i, a in enumerate(aggregates)])
        ax.barh(y, widths, left=left, color=color, edgecolor="white",
                linewidth=0.5, label=cat)
        for i, w in enumerate(widths):
            if w > 6:
                ax.text(left[i] + w / 2, i, f"{w:.0f}%",
                        va="center", ha="center", fontsize=8.5,
                        color="white" if cat != "Loop overhead" else "#333",
                        weight="bold")
        left += widths
    for i, t in enumerate(totals):
        ax.text(101, i, f"{t:.1f} μs", va="center", ha="left",
                fontsize=10, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(kernels)
    ax.set_xlabel("share of kernel time  [%]")
    ax.set_ylabel("")
    ax.set_title("PL kernel per-stage breakdown — normalized (csynth estimates)",
                 fontsize=12, weight="bold", pad=10)
    ax.set_xlim(0, 118)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, ncol=3, framealpha=0.95,
              frameon=True, edgecolor="#ccc")
    out2 = OUT_DIR / "kernel_stage_breakdown_pct.png"
    fig.savefig(out2, bbox_inches="tight")
    fig.savefig(OUT_DIR / "kernel_stage_breakdown_pct.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    plot_totals()
    plot_breakdown()
