#!/usr/bin/env python3
"""PL-stream vs AIE+PL-hybrid comparison, measured on VCK190 (xcvc1902).

Two configurations of the full anomaly model:
- all-PL : entire pipeline (incl. attention) in PL fabric.
- hybrid : attention offloaded to the AIE array; PL runs embed/pairwise/
           build/lorentz/autoencoder (pairwise MLP pipelined).

All numbers MEASURED this campaign:
- latency/throughput: host CU start->done, ert_polling (N=10 x 20 iters).
- resources: Vivado full_util_routed.rpt (full device, post-route).
- accuracy: HW output vs golden (all-PL exact; hybrid +17% int16 AIE-attn quant).
- power: VCK190 System-Controller INA226 rails (~2.1 W device, static-dominated).

Outputs (figs/): pl_vs_hybrid_comparison, pl_vs_hybrid_table
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
PL_C  = "#d62728"   # all-PL
HYB_C = "#1f77b4"   # hybrid
NEUTRAL = "#7f7f7f"; GOOD = "#1a8a2e"

DSP_AVAIL, LUT_AVAIL, TILES_AVAIL = 1968, 899840, 400

# ---- the two measured configs --------------------------------------------
PL  = dict(label="all-PL",  c=PL_C,  lat=2.09, thr=478,  dsp=1055, lut=204895, tiles=0,  err=0.0,  clk=80)
HYB = dict(label="hybrid",  c=HYB_C, lat=0.894, thr=1118, dsp=1680, lut=120809, tiles=78, err=16.9, clk=100)
CFG = [PL, HYB]


def _pair(ax, vals, title, ylabel, fmt, ymax_mul=1.30):
    xs = np.arange(2); cols = [c["c"] for c in CFG]
    ax.bar(xs, vals, color=cols, width=0.6)
    ymax = max(vals) * ymax_mul; ax.set_ylim(0, ymax)
    for x, v in zip(xs, vals):
        ax.text(x, v + ymax*0.015, fmt.format(v), ha="center", va="bottom",
                fontsize=11, weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([c["label"] for c in CFG], fontsize=10.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11.5, weight="bold", pad=8)
    return ymax


def fig_comparison(path):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.7))

    # (a) latency
    ax = axes[0]
    _pair(ax, [c["lat"] for c in CFG], "Per-event latency", "ms / event", "{:.2f}")
    ax.text(0.5, -0.20, f"hybrid {PL['lat']/HYB['lat']:.2f}× faster",
            transform=ax.transAxes, ha="center", va="top", color=GOOD,
            weight="bold", fontsize=10)

    # (b) throughput
    ax = axes[1]
    _pair(ax, [c["thr"] for c in CFG], "Throughput", "events / s", "{:.0f}")
    ax.text(0.5, -0.20, f"{PL['thr']}→{HYB['thr']} ev/s",
            transform=ax.transAxes, ha="center", va="top", color=GOOD,
            weight="bold", fontsize=10)

    # (c) resource utilization (% of device), grouped
    ax = axes[2]
    res = ["PL DSP", "PL LUT", "AIE tiles"]
    pl_pct  = [100*PL["dsp"]/DSP_AVAIL, 100*PL["lut"]/LUT_AVAIL, 100*PL["tiles"]/TILES_AVAIL]
    hyb_pct = [100*HYB["dsp"]/DSP_AVAIL, 100*HYB["lut"]/LUT_AVAIL, 100*HYB["tiles"]/TILES_AVAIL]
    raw = {"all-PL": [f"{PL['dsp']:,}", f"{PL['lut']:,}", f"{PL['tiles']}"],
           "hybrid": [f"{HYB['dsp']:,}", f"{HYB['lut']:,}", f"{HYB['tiles']}"]}
    x = np.arange(len(res)); w = 0.38
    b1 = ax.bar(x-w/2, pl_pct,  w, color=PL_C,  label="all-PL")
    b2 = ax.bar(x+w/2, hyb_pct, w, color=HYB_C, label="hybrid")
    ymax = max(pl_pct+hyb_pct)*1.25; ax.set_ylim(0, ymax)
    for xi, v, lab in zip(x-w/2, pl_pct, raw["all-PL"]):
        ax.text(xi, v+ymax*0.015, lab, ha="center", va="bottom", fontsize=8.2, weight="bold", color=PL_C)
    for xi, v, lab in zip(x+w/2, hyb_pct, raw["hybrid"]):
        ax.text(xi, v+ymax*0.015, lab, ha="center", va="bottom", fontsize=8.2, weight="bold", color=HYB_C)
    ax.set_xticks(x); ax.set_xticklabels(res, fontsize=9.5)
    ax.set_ylabel("% of VC1902 device"); ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title("Resource usage", fontsize=11.5, weight="bold", pad=8)

    # (d) accuracy
    ax = axes[3]
    ym = _pair(ax, [c["err"] for c in CFG], "Accuracy error vs golden",
               "% error (MSE)", "{:.1f}%", ymax_mul=1.5)
    ax.text(0.5, -0.20, "hybrid: int16 AIE-attn quant",
            transform=ax.transAxes, ha="center", va="top", color=NEUTRAL,
            weight="bold", fontsize=10)

    fig.suptitle("PL-stream vs AIE+PL hybrid — full anomaly model on VCK190 (xcvc1902)",
                 y=1.04, fontsize=13, weight="bold")
    fig.text(0.5, -0.02,
             "Device power ~2.1 W for both (INA226, static-dominated).  "
             "Hybrid: attention on 78 AIE tiles + pipelined pairwise MLP; all-PL: everything in fabric.",
             ha="center", fontsize=8.4, color=NEUTRAL)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_table(path):
    rows = [
        ["Metric", "all-PL", "hybrid"],
        ["Attention on", "PL fabric", "AIE (78 tiles)"],
        ["Kernel clock", "80 MHz", "100 MHz"],
        ["Latency / event", "2.09 ms", "0.894 ms"],
        ["Throughput", "478 ev/s", "1118 ev/s"],
        ["PL LUT", "204,895 (23%)", "120,809 (13%)"],
        ["PL DSP", "1,055 (54%)", "1,680 (85%)"],
        ["AIE tiles", "0", "78 (19.5%)"],
        ["Accuracy (MSE err)", "exact (golden)", "+17%"],
        ["Device power", "~2.1 W", "~2.1 W"],
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.0)); ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 1.75)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#efefef"); cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(weight="bold")
        if c == 1 and r > 0:
            cell.set_facecolor("#eaf1f8")   # PL tint
        if c == 2 and r > 0:
            cell.set_facecolor("#faecec")   # hybrid tint
    ax.set_title("PL-stream vs AIE+PL hybrid — measured on VCK190 (xcvc1902)",
                 fontsize=12.5, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figs"); os.makedirs(out, exist_ok=True)
    print("output dir:", out)
    fig_comparison(os.path.join(out, "pl_vs_hybrid_comparison"))
    fig_table(os.path.join(out, "pl_vs_hybrid_table"))
    print("Done.")
