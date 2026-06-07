#!/usr/bin/env python3
"""Render per-tile AIE profile charts in the aie_vs_pl_steady style.

Inputs:  scripts/aie_profile_per_tile.json
Outputs: figs/aie_per_tile_busy.png         (busy% per tile, grouped by block)
         figs/aie_per_tile_breakdown.png    (stacked bars, function categories)
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

# ----------------------------------------------------------------------
# Categorize tiles by block + role for layout
# ----------------------------------------------------------------------
BLOCK_OF = {"obj": "Obj", "cand": "Cand", "cross": "Cross"}

ROLE_PALETTE = {
    "head_pre":  "#3a6fb0",   # blue
    "head_post": "#c0392b",   # red
    "post":      "#2e8b57",   # green
}

CAT_PALETTE = {
    "softmax (expf)":       "#c0392b",
    "scalar-float helpers": "#e67e22",
    "layernorm + sqrt":     "#9467bd",
    "fixed↔float convert":  "#f1c40f",
    "vector mmul":          "#2e8b57",
    "memcpy / memset":      "#17becf",
    "kernel body / glue":   "#7f8c8d",
    "main / init":          "#bdc3c7",
    "other":                "#34495e",
}
CAT_ORDER = list(CAT_PALETTE.keys())

# Top-of-stack compute primitives for the INCLUSIVE breakdown view.
# We attribute each tile's active cycles to whichever of these the cycles
# fell under (using cyc_func_desc). The remainder is "kernel body / other".
INCLUSIVE_BUCKETS = [
    ("softmax (expf)",      ["expf"]),
    ("layernorm + sqrt",    ["layernorm_row", "float_sqrtf", "softfloat_approxRecip32_1"]),
    ("vector mmul",         ["gemm_tile<12,",   # any gemm_tile instantiation
                             "gemm_tile<4,",
                             "gemm_tile<3,",
                             "gemm_tile"]),
    ("fixed↔float convert", ["f32_to_i32_r_minMag", "i32_to_f32"]),
]


def kernel_label(k):
    """Short readable label."""
    return (k.replace("_attn_head_", "_h")
              .replace("_L0", "")
              .replace("_post_", "_post-"))


def role_of(kernel):
    k = kernel.lower()
    if "head_pre" in k:
        return "head_pre"
    if "head_post" in k:
        return "head_post"
    return "post"


def block_of(kernel):
    for prefix in ("obj", "cand", "cross"):
        if kernel.startswith(prefix):
            return BLOCK_OF[prefix]
    return "?"


# ----------------------------------------------------------------------
# Plot 1: per-tile busy %  (grouped by block × role)
# ----------------------------------------------------------------------
def plot_busy():
    rows = []
    for t in DATA:
        if t["busy_pct"] is None:
            continue
        rows.append({
            "tile":   f"{t['col']}_{t['row']}",
            "kernel": t["kernel"],
            "block":  block_of(t["kernel"]),
            "role":   role_of(t["kernel"]),
            "busy":   t["busy_pct"],
            "pm":     t["pm_size"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["block", "role", "kernel"])

    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    x = np.arange(len(df))
    colors = [ROLE_PALETTE[r] for r in df["role"]]
    bars = ax.bar(x, df["busy"], width=0.78, color=colors, edgecolor="none")

    for i, v in enumerate(df["busy"]):
        ax.text(x[i], v + 1.2, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=8, weight="bold", color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([kernel_label(k) for k in df["kernel"]],
                       rotation=55, ha="right", fontsize=8.5)
    ax.set_ylabel("tile busy  [% of total sim cycles]")
    ax.set_ylim(0, 115)
    ax.set_title(
        "AIE per-tile utilization — obj head_post tiles are pipeline-saturating",
        fontsize=12, weight="bold", pad=10)

    # block dividers
    prev_block = None
    for i, b in enumerate(df["block"]):
        if b != prev_block and prev_block is not None:
            ax.axvline(i - 0.5, color="#aaa", linestyle="-", linewidth=0.8, alpha=0.7)
        prev_block = b
    # role legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=ROLE_PALETTE[r])
                      for r in ROLE_PALETTE]
    ax.legend(legend_handles, ["head_pre (QKV gemm)", "head_post (softmax)",
                               "post (proj + LN + FFN)"],
              loc="upper right", fontsize=9.5, framealpha=0.95,
              edgecolor="#ccc")
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(OUT / "aie_per_tile_busy.png", bbox_inches="tight")
    fig.savefig(OUT / "aie_per_tile_busy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'aie_per_tile_busy.png'}")


# ----------------------------------------------------------------------
# Plot 2: per-tile function-category breakdown (stacked bars)
# ----------------------------------------------------------------------
def aggregate_categories(tile):
    """Return dict category -> percent of active cycles for this tile.

    Uses *inclusive* cycle counts (cyc_func_desc) for a small set of
    top-of-stack compute primitives (expf, layernorm_row, gemm_tile, ...),
    so the bar segments represent what the tile is *actually doing* (not
    leaked across the call hierarchy). The unaccounted remainder is
    'kernel body / glue'.
    """
    cats = defaultdict(float)
    used = set()
    for bucket_name, name_prefixes in INCLUSIVE_BUCKETS:
        for fn in tile["functions"]:
            fname = fn["name"]
            if any(fname.startswith(p) for p in name_prefixes):
                if fname in used:
                    continue
                used.add(fname)
                cats[bucket_name] += fn["pct_func_desc"]
    # Whatever's left of the active cycles
    total = sum(cats.values())
    rest = max(0.0, 100.0 - total)
    cats["kernel body / glue"] = rest
    for c in CAT_ORDER:
        cats.setdefault(c, 0.0)
    return dict(cats)


def plot_breakdown():
    rows = []
    for t in DATA:
        if t["busy_pct"] is None:
            continue
        cats = aggregate_categories(t)
        rows.append({
            "tile":   f"{t['col']}_{t['row']}",
            "kernel": t["kernel"],
            "block":  block_of(t["kernel"]),
            "role":   role_of(t["kernel"]),
            "busy":   t["busy_pct"],
            "cats":   cats,
        })
    rows.sort(key=lambda r: (r["block"], r["role"], r["kernel"]))

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    # Scale each category by busy% so the stacked bar height = % of total sim
    for cat in CAT_ORDER:
        widths = np.array([
            r["cats"].get(cat, 0.0) * r["busy"] / 100.0 for r in rows
        ])
        ax.bar(x, widths, bottom=bottom, color=CAT_PALETTE[cat],
               edgecolor="none", width=0.78, label=cat)
        for i, w in enumerate(widths):
            if w >= 7:
                ax.text(x[i], bottom[i] + w/2, f"{w:.0f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if cat != "main / init" else "#222",
                        weight="bold")
        bottom += widths

    # idle (= 100 - busy) as a light gray cap so heights all hit 100
    idle = np.array([100.0 - r["busy"] for r in rows])
    ax.bar(x, idle, bottom=bottom, color="#ecf0f1", edgecolor="none",
           width=0.78, label="idle / stream stall")

    ax.set_xticks(x)
    ax.set_xticklabels([kernel_label(r["kernel"]) for r in rows],
                       rotation=55, ha="right", fontsize=8.0)
    ax.set_ylabel("% of total sim cycles  (5 M @ 1.25 GHz = 4 ms)")
    ax.set_ylim(0, 108)
    ax.set_title(
        "AIE per-tile time breakdown — head_post tiles saturated by scalar softmax",
        fontsize=12, weight="bold", pad=10)

    # block dividers
    prev_block = None
    for i, r in enumerate(rows):
        if r["block"] != prev_block and prev_block is not None:
            ax.axvline(i - 0.5, color="#aaa", linestyle="-", linewidth=0.8, alpha=0.7)
        prev_block = r["block"]

    ax.legend(loc="upper right", fontsize=8.5, ncol=2,
              framealpha=0.95, edgecolor="#ccc")
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(OUT / "aie_per_tile_breakdown.png", bbox_inches="tight")
    fig.savefig(OUT / "aie_per_tile_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'aie_per_tile_breakdown.png'}")


if __name__ == "__main__":
    plot_busy()
    plot_breakdown()
