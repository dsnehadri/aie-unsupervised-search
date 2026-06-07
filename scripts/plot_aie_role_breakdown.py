#!/usr/bin/env python3
"""Plot per-(block × role) cycle breakdown showing where the AIE sim
spends time, separating real compute from stream-stall.

Output: figs/aie_role_breakdown.png
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
DATA = json.loads((REPO / "scripts/aie_profile_per_tile.json").read_text())
OUT  = REPO / "figs"
OUT.mkdir(exist_ok=True)

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

PALETTE = {
    "softmax (expf)":      "#c0392b",
    "layernorm + sqrt":    "#9467bd",
    "vector mmul":         "#2e8b57",
    "convert / other":     "#f1c40f",
    "kernel body / rest":  "#e67e22",
    "stream-stall":        "#bdc3c7",
}

PRIM_MAP = {
    "softmax (expf)":    lambda n: n.startswith("expf"),
    "layernorm + sqrt":  lambda n: (n.startswith("layernorm_row")
                                    or n.startswith("float_sqrtf")
                                    or n.startswith("softfloat_approxRecip32_1")
                                    or n.startswith("f32_div")),
    "vector mmul":       lambda n: n.startswith("gemm_tile"),
    "convert / other":   lambda n: (n.startswith("f32_to_i32")
                                    or n.startswith("i32_to_f32")),
}

ROLES  = ["head_pre", "head_post", "post"]
BLOCKS = ["obj", "cross", "cand"]


def role_of(k):
    if "head_pre" in k:  return "head_pre"
    if "head_post" in k: return "head_post"
    return "post"


def block_of(k):
    for b in BLOCKS:
        if k.startswith(b): return b
    return "?"


def aggregate():
    """Return {(block,role): {category: cycles}} normalized to total active."""
    bucket = defaultdict(lambda: defaultdict(int))
    total_active = 0
    for t in DATA:
        bk, rl = block_of(t["kernel"]), role_of(t["kernel"])
        active = t["report_cycles"] or 0
        total_active += active
        # kernel work cycles = kernel function's inclusive cycles
        kwork = 0
        for fn in t["functions"]:
            if t["kernel"] in fn["name"]:
                kwork = fn["cyc_func_desc"]
                break
        stall = max(0, active - kwork)
        bucket[(bk, rl)]["stream-stall"] += stall

        used = 0
        for cat, pred in PRIM_MAP.items():
            for fn in t["functions"]:
                if pred(fn["name"]):
                    bucket[(bk, rl)][cat] += fn["cyc_func_desc"]
                    used += fn["cyc_func_desc"]
        # "kernel body / rest" = work cycles not in any primitive bucket
        bucket[(bk, rl)]["kernel body / rest"] += max(0, kwork - used)

    return bucket, total_active


def main():
    bucket, total_active = aggregate()

    # Build matrix: rows = (block, role), columns = categories
    cats = ["softmax (expf)", "layernorm + sqrt", "vector mmul",
            "convert / other", "kernel body / rest", "stream-stall"]
    labels = []
    matrix = []
    for bk in BLOCKS:
        for rl in ROLES:
            labels.append(f"{bk}\n{rl}")
            row = [bucket[(bk, rl)].get(c, 0) for c in cats]
            matrix.append(row)
    M = np.array(matrix, dtype=float) / total_active * 100.0   # % of total active

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for j, cat in enumerate(cats):
        widths = M[:, j]
        ax.bar(x, widths, bottom=bottom, color=PALETTE[cat],
               edgecolor="none", width=0.66, label=cat)
        for i, w in enumerate(widths):
            if w >= 1.0:
                ax.text(x[i], bottom[i] + w/2, f"{w:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if cat not in ("convert / other", "stream-stall")
                              else "#222",
                        weight="bold")
        bottom += widths

    # block dividers
    for i in (3, 6):
        ax.axvline(i - 0.5, color="#aaa", linestyle="-", linewidth=1.0, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("% of total AIE-array compute cycles\n(sum across all 39 tiles)")
    ymax = bottom.max() * 1.06
    ax.set_ylim(0, ymax)

    # Per-block bracket annotations on top
    for i, blk in enumerate(BLOCKS):
        x0, x1 = i*3, i*3+2
        blk_pct = M[i*3:i*3+3].sum()
        ax.text((x0+x1)/2, ymax * 0.96,
                f"{blk}: {blk_pct:.1f}% of all cycles",
                ha="center", va="top", fontsize=10.5, weight="bold",
                color="#222")

    ax.set_title("Where AIE compute time goes — softmax in head_post tiles dominates",
                 fontsize=12.5, weight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=9.5, framealpha=0.95, edgecolor="#ccc",
              ncol=6, frameon=True)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / "aie_role_breakdown.png", bbox_inches="tight", dpi=170)
    fig.savefig(OUT / "aie_role_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'aie_role_breakdown.png'}")

    # Print headline numbers for the user
    print(f"\nTotal active core-cycles: {total_active:,}")
    softmax_cyc = sum(bucket[k].get("softmax (expf)", 0) for k in bucket)
    ln_cyc      = sum(bucket[k].get("layernorm + sqrt", 0) for k in bucket)
    mmul_cyc    = sum(bucket[k].get("vector mmul", 0) for k in bucket)
    stall_cyc   = sum(bucket[k].get("stream-stall", 0) for k in bucket)
    print(f"  softmax (expf+descendants): {softmax_cyc/total_active*100:5.1f}%")
    print(f"  layernorm + sqrt          : {ln_cyc     /total_active*100:5.1f}%")
    print(f"  vector mmul               : {mmul_cyc   /total_active*100:5.1f}%")
    print(f"  stream-stall              : {stall_cyc  /total_active*100:5.1f}%")


if __name__ == "__main__":
    main()
