#!/usr/bin/env python3
"""Final summary plots: all-PL vs final hybrid, stage breakdown, attn scaling.

MEASURED on VCK190 (host_aie_timed) unless noted:
- all-PL   : 2.09 ms/event single-event latency, 478 ev/s throughput.
- hybrid   : 0.916 ms single-event latency (N=1), 7549 ev/s throughput (N=100,
             cross-event pipelined). attention on 78 AIE tiles, pairwise pipelined.
- attention blocks standalone: obj 4738 ev/s @0.211 ms (bridge_solo, 13 tiles),
             cand 14808 ev/s @0.068 ms (cand_emu, 13 tiles).
- PL per-stage times from HLS csynth (cycles x10 ns @100 MHz).
- cross-attn AIE time estimated from aiesim ratio (no standalone HW vehicle).

Outputs (figs/): final_hybrid_vs_pl, final_stage_breakdown, final_attn_scaling
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
PL_C, HYB_C, AIE_C, GOOD, NEUTRAL = "#d62728", "#1f77b4", "#1f77b4", "#1a8a2e", "#7f7f7f"


# ============================================================ Plot 1
def fig_summary(path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.6))
    cfg = ["all-PL\n(0 AIE tiles)", "hybrid\n(78 AIE tiles)"]; cols = [PL_C, HYB_C]

    lat = [2.09, 0.916]
    a1.bar(cfg, lat, color=cols, width=0.55)
    a1.set_ylim(0, max(lat)*1.2)
    for i, v in enumerate(lat):
        a1.text(i, v+max(lat)*0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=12, weight="bold")
    a1.set_ylabel("latency per event  [ms]")
    a1.set_title("Latency", fontsize=12, weight="bold")

    thr = [478, 7549]
    a2.bar(cfg, thr, color=cols, width=0.55)
    a2.set_ylim(0, max(thr)*1.2)
    for i, v in enumerate(thr):
        a2.text(i, v+max(thr)*0.02, f"{v:,}", ha="center", va="bottom", fontsize=12, weight="bold")
    a2.set_ylabel("throughput  [events / s]")
    a2.set_title("Throughput", fontsize=12, weight="bold")

    fig.suptitle("Final hybrid vs all-PL (VCK190)", y=1.02, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


# ============================================================ Plot 2
def fig_stage_breakdown(path):
    # (stage, microseconds/event, engine)  -- AIE=attention, PL=rest
    stages = [
        ("Object attention  (×2 layers)", 422, "AIE"),
        ("Cross attention  (×2, est.)",   200, "AIE"),
        ("Candidate attention  (×2)",     136, "AIE"),
        ("Embed FFN",                      47, "PL"),
        ("Remask / Lorentz / IO",          36, "PL"),
        ("Pairwise MLP (wij, pipelined)",  26, "PL"),
        ("Candidate build (×2)",           18, "PL"),
        ("Autoencoder",                    13, "PL"),
    ]
    stages = sorted(stages, key=lambda s: s[1])   # ascending -> largest on top
    aie_tot = sum(s[1] for s in stages if s[2] == "AIE")
    pl_tot  = sum(s[1] for s in stages if s[2] == "PL")
    tot = aie_tot + pl_tot

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    y = np.arange(len(stages))
    cols = [AIE_C if s[2] == "AIE" else PL_C for s in stages]
    vals = [s[1] for s in stages]
    ax.barh(y, vals, color=cols, height=0.66)
    for yi, s in zip(y, stages):
        ax.text(s[1] + tot*0.008, yi, f"{s[1]} µs", va="center", ha="left",
                fontsize=10, weight="bold", color=(AIE_C if s[2]=="AIE" else PL_C))
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in stages], fontsize=10)
    ax.set_xlim(0, max(vals)*1.16)
    ax.set_xlabel("time per event  [µs]")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AIE_C, label="AIE attention"),
                       Patch(color=PL_C, label="PL fabric")],
              loc="lower right", fontsize=11, frameon=True)
    ax.set_title("Per-event time by pipeline stage", fontsize=12.5, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


# ============================================================ Plot 3
def fig_attn_scaling(path):
    """Attention blocks ONLY: latency vs throughput. AIE throughput scales with
    tiles (13 tiles/instance, up to ~30 in the 400-tile array); PL is fabric-bound."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    R = np.array([1, 2, 4, 8, 16, 30])            # replicas (30 = 390/400 tiles)

    # AIE obj (measured single instance, replicate over tiles)
    ax.plot([0.211]*len(R), 4738*R, "-o", color="#1f77b4", lw=2, ms=7,
            label="AIE object-attn  (13 tiles / instance)")
    # AIE cand
    ax.plot([0.068]*len(R), 14808*R, "-s", color="#9ecae1", lw=2, ms=7,
            label="AIE candidate-attn (13 tiles / instance)")

    # PL attention reference (HLS csynth, @80 MHz all-PL clock; DSP-bound)
    pl_lat, pl_thr = 0.158, 6334   # measured runtime (pl_attn)
    ax.plot([pl_lat], [pl_thr], "D", color=PL_C, ms=11, label="PL object-attn (fabric)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.04, 0.6); ax.set_ylim(2e3, 6e5)
    ax.set_xlabel("per-event latency  [ms]  (log)")
    ax.set_ylabel("throughput  [events / s]  (log)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title("Attention blocks: latency vs throughput (AIE scales with tile count)",
                 fontsize=12, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


# ============================================================ Plot 3b
# PL=red, AIE=blue (engine colors)
AIE_C, PL_C = "#1f77b4", "#d62728"
# Object-attention scaling. ALL points MEASURED on HW from ONE self-consistent
# build (obj16_emu, n_inst swept in a single process). 13 AIE tiles per instance.
OBJ_MEAS = {13:  dict(thr=4647.7,  lat=0.2152),   # 1x
            26:  dict(thr=8701.3,  lat=0.2298),   # 2x
            52:  dict(thr=15225.9, lat=0.2627),   # 4x
            104: dict(thr=24965.7, lat=0.3204),   # 8x
            208: dict(thr=36659.3, lat=0.4365)}   # 16x (max fit)
PL_OBJ = dict(thr=6334.0, lat=0.158)             # MEASURED runtime (pl_attn vehicle)

# per-block AIE vs PL, both MEASURED on HW (single instance)
BLOCK_MEAS = {
    "obj":   dict(aie_thr=4738,  aie_lat=0.211, pl_thr=6334,  pl_lat=0.158),
    "cand":  dict(aie_thr=14808, aie_lat=0.068, pl_thr=25458, pl_lat=0.039),
    "cross": dict(aie_thr=5212,  aie_lat=0.192, pl_thr=7307,  pl_lat=0.137),
}


def fig_block_compare(path):
    blocks = ["obj", "cand", "cross"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
    x = np.arange(len(blocks)); w = 0.38
    # throughput
    aie = [BLOCK_MEAS[b]["aie_thr"] for b in blocks]
    pl  = [BLOCK_MEAS[b]["pl_thr"]  for b in blocks]
    a1.bar(x-w/2, aie, w, color=AIE_C, label="AIE")
    a1.bar(x+w/2, pl,  w, color=PL_C,  label="PL")
    ymx=max(aie+pl)*1.18; a1.set_ylim(0,ymx)
    for xi,v in zip(x-w/2,aie): a1.text(xi,v+ymx*0.01,f"{v:,}",ha="center",va="bottom",fontsize=8.5,weight="bold",color=AIE_C)
    for xi,v in zip(x+w/2,pl):  a1.text(xi,v+ymx*0.01,f"{v:,}",ha="center",va="bottom",fontsize=8.5,weight="bold",color=PL_C)
    a1.set_xticks(x); a1.set_xticklabels(blocks); a1.set_ylabel("throughput  [events / s]")
    a1.legend(fontsize=10); a1.set_title("Throughput (single instance)", fontsize=11.5, weight="bold")
    # latency
    aiel=[BLOCK_MEAS[b]["aie_lat"] for b in blocks]
    pll =[BLOCK_MEAS[b]["pl_lat"]  for b in blocks]
    a2.bar(x-w/2, aiel, w, color=AIE_C, label="AIE")
    a2.bar(x+w/2, pll,  w, color=PL_C,  label="PL")
    ymx2=max(aiel+pll)*1.18; a2.set_ylim(0,ymx2)
    for xi,v in zip(x-w/2,aiel): a2.text(xi,v+ymx2*0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=8.5,weight="bold",color=AIE_C)
    for xi,v in zip(x+w/2,pll):  a2.text(xi,v+ymx2*0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=8.5,weight="bold",color=PL_C)
    a2.set_xticks(x); a2.set_xticklabels(blocks); a2.set_ylabel("latency  [ms / event]")
    a2.legend(fontsize=10); a2.set_title("Latency (single instance)", fontsize=11.5, weight="bold")
    fig.suptitle("Attention blocks: AIE vs PL (single instance, measured on VCK190)",
                 y=1.03, fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")
def fig_obj_throughput(path):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    mt = sorted(OBJ_MEAS)
    # ideal linear scaling from the 1x measured rate (upper bound)
    rate = OBJ_MEAS[13]["thr"] / 13.0
    ideal_x = np.array([0] + mt)
    ax.plot(ideal_x, rate*ideal_x, "--", color=AIE_C, lw=1.8, alpha=0.5,
            label="AIE obj (ideal linear)")
    # measured AIE points (connected -> real DMA-limited curve)
    ax.plot(mt, [OBJ_MEAS[t]["thr"] for t in mt], "-o", color=AIE_C, lw=2, ms=10,
            label="AIE obj (measured)")
    # PL reference
    ax.axhline(PL_OBJ["thr"], color=PL_C, ls="--", lw=2, label="PL obj (measured)")
    ax.set_xlim(0, 220); ax.set_ylim(0, rate*208*1.08)
    ax.set_xlabel("AIE tiles used")
    ax.set_ylabel("throughput  [events / s]")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Object attention: throughput vs AIE tiles", fontsize=12.5, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_obj_latency(path):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    mt = sorted(OBJ_MEAS)
    ax.plot(mt, [OBJ_MEAS[t]["lat"] for t in mt], "-o", color=AIE_C, lw=2, ms=10,
            label="AIE obj (measured)")
    ax.axhline(PL_OBJ["lat"], color=PL_C, ls="--", lw=2, label="PL obj (measured)")
    ax.set_xlim(0, 220); ax.set_ylim(0, 0.48)
    ax.set_xlabel("AIE tiles used")
    ax.set_ylabel("latency  [ms / event]")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Object attention: latency vs AIE tiles", fontsize=12.5, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight"); fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figs"); os.makedirs(out, exist_ok=True)
    print("output dir:", out)
    fig_summary(os.path.join(out, "final_hybrid_vs_pl"))
    fig_stage_breakdown(os.path.join(out, "final_stage_breakdown"))
    fig_attn_scaling(os.path.join(out, "final_attn_scaling"))
    fig_obj_throughput(os.path.join(out, "final_attn_obj_throughput"))
    fig_obj_latency(os.path.join(out, "final_attn_obj_latency"))
    fig_block_compare(os.path.join(out, "final_attn_block_compare"))
    print("Done.")
