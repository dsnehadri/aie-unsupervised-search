#!/usr/bin/env python3
"""AIE-usage scaling charts: impact of moving attention blocks PL -> AIE.

The model has 6 attention subgraphs (obj/cand/cross x 2 layers). Each can run
in PL fabric or be offloaded to the AIE array. This sweeps "how much AIE the
algorithm uses" and shows the impact on PL resources, AIE tiles, latency,
throughput, power and accuracy.

DATA PROVENANCE
- AIE tiles / PLIO per #blocks: distinct CR(x,y) in the aiecompiler mapping
  reports of the obj0 / oc / l0 / full builds (Work_hw_{obj0,oc,l0,full}).
  Each attention subgraph = 13 tiles -> 0/13/26/39/78 for 0/1/2/3/6 blocks.
- PL fabric (full device, post-route full_util_routed.rpt) and end-to-end HW
  latency/throughput/power/accuracy are MEASURED for the two configs that
  compute the *full* model: all-PL (0 AIE) and all-AIE (78 AIE).
- The 1/2/3-block builds (obj0/oc/l0) are partial-pipeline isolation builds,
  so their PL resources are NOT full-model offload points; only their AIE tile
  counts are used (those are exact regardless of pipeline completeness).
- PL-DSP between the two full-model endpoints is shown as a linear projection
  (dashed); only k=0 and k=6 were built as full models.

Outputs (figs/): aie_scaling_tiles, aie_scaling_resources, aie_scaling_impact
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)
import matplotlib.font_manager as _fm
if any("ontserrat" in (f or "").lower() for f in _fm.findSystemFonts()):
    plt.rcParams["font.family"] = "Montserrat"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "patch.linewidth": 0.0,
})
PL_COLOR, AIE_COLOR, NEUTRAL = "#d62728", "#1f77b4", "#7f7f7f"
GOOD, BAD = "#1a8a2e", "#a31515"

# VC1902 device budget
AIE_AVAIL = 400
PL_AVAIL = {"LUT": 899840, "FF": 1799680, "DSP": 1968, "BRAM": 967}

# ---- scaling axis: # attention blocks offloaded to AIE -----------------
# tiles/PLIO measured from mapping reports (obj0=1, oc=2, l0=3, full=6 blocks)
BLOCKS_K   = [0, 1, 2, 3, 6]
AIE_TILES  = [0, 13, 26, 39, 78]      # exact, 13 tiles per attention subgraph
AIE_PLIO   = [0, 6, 8, 11, 22]        # SHIM stream channels (obj has +wij)
MEAS_FULL  = {0, 6}                    # which k were built as the FULL model

# ---- full-model endpoints (MEASURED on HW this session) ----------------
# all-PL (0 AIE) and all-AIE (78 AIE); device = xcvc1902, VCK190
END = {
    0: dict(name="all-PL\n(0 AIE tiles)",
            LUT=204895, FF=201902, DSP=1055, BRAM=12, tiles=0,
            lat_ms=2.09, thr=478, power_W=2.1, clk=80,
            mse=0.52134, err_pct=0.0),
    6: dict(name="all-AIE\n(78 AIE tiles)",
            LUT=70607, FF=97191, DSP=314, BRAM=13.5, tiles=78,
            lat_ms=1.81, thr=551, power_W=2.1, clk=100,
            mse=0.60988, err_pct=17.0),
}

# ---- per-block AIE int16 quantization error (x86sim valid rows, tol 0.5) -
PERBLOCK_ERR = {"obj (self)": 0.027, "cand (self)": 0.011, "cross": 0.064}

# ---- MEASURED standalone attention-block throughput (HW, golden-fed) -----
# repeated single-event invocations of the isolated AIE attention subgraph
# (obj=bridge_solo, cand=cand_emu); 13 AIE tiles + a small PL streaming bridge.
THR_BLOCK = {"obj\nself-attn": 4738, "cand\nself-attn": 14808}   # events/s
LAT_BLOCK = {"obj\nself-attn": 0.211, "cand\nself-attn": 0.0675} # ms/event
THR_FULL  = 551     # full model (all 6 attn serial + PL pipeline), 78 tiles

# ---- replica resource cost (MEASURED) -> event-parallel ceiling ----------
# full model = PL pipeline (embed/build/lorentz/AE) + all 6 attn on AIE.
# attn block = one attention subgraph + its PL streaming bridge (cand_emu).
REPLICA = {
    "full model\n(PL + 6 attn)":  dict(tiles=78, dsp=314, lut=66682, thr=551),
    "1 attn block\n(AIE+bridge)": dict(tiles=13, dsp=0,   lut=2737,  thr=4738),
}


def _labels(ax, fmt, ymax=None, fs=9.5, color=None):
    if ymax is None: ymax = ax.get_ylim()[1]
    pad = ymax * 0.014
    for cont in ax.containers:
        for p in cont:
            h = p.get_height()
            if not np.isfinite(h) or h == 0: continue
            ax.text(p.get_x()+p.get_width()/2, h+pad, fmt.format(h),
                    ha="center", va="bottom", fontsize=fs, weight="bold",
                    color=color or "black")


def fig_tiles(path):
    """AIE tiles + PLIO channels vs number of attention blocks offloaded."""
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    k = np.array(BLOCKS_K)
    ax.plot(k, AIE_TILES, "-o", color=AIE_COLOR, lw=2.4, ms=8,
            label="AIE compute tiles  (13 / block)")
    # ideal-linear reference (13/block) across 0..6
    kfull = np.arange(0, 7)
    ax.plot(kfull, 13*kfull, ":", color=AIE_COLOR, lw=1.2, alpha=0.6)
    for xx, yy in zip(k, AIE_TILES):
        ax.annotate(f"{yy}", (xx, yy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, weight="bold",
                    color=AIE_COLOR)
    ax.set_xlabel("attention blocks running on AIE  (of 6)")
    ax.set_ylabel("AIE compute tiles", color=AIE_COLOR)
    ax.tick_params(axis="y", labelcolor=AIE_COLOR)
    ax.set_xticks(kfull)
    ax.set_ylim(0, 90)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(k, AIE_PLIO, "-s", color=PL_COLOR, lw=2.0, ms=7,
             label="PL↔AIE stream channels (PLIO)")
    ax2.set_ylabel("PL↔AIE PLIO stream channels", color=PL_COLOR)
    ax2.tick_params(axis="y", labelcolor=PL_COLOR)
    ax2.set_ylim(0, 30)
    ax2.grid(False)

    # % of array on the right side annotation for full
    ax.axhline(AIE_AVAIL, color=NEUTRAL, ls="--", lw=0.8)
    ax.text(0.05, 78+1.5, f"all-AIE = 78/{AIE_AVAIL} tiles "
            f"({100*78/AIE_AVAIL:.1f}% of array)",
            fontsize=9, color=NEUTRAL, weight="bold")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lab1+lab2, loc="upper left", fontsize=9.5)
    ax.set_title("Scaling AIE usage: tile & stream cost is linear in blocks offloaded",
                 fontsize=12, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_resources(path):
    """PL fabric freed by offloading + projected DSP vs blocks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.7))

    # ---- LEFT: full-model PL fabric, all-PL vs all-AIE (measured) -------
    res = ["LUT", "FF", "DSP"]
    rows = []
    for k in (0, 6):
        for r in res:
            rows.append({"cfg": END[k]["name"], "resource": r,
                         "pct": 100*END[k][r]/PL_AVAIL[r], "raw": END[k][r]})
    df = pd.DataFrame(rows)
    pal = {"LUT": "#3a6fb0", "FF": "#2ca02c", "DSP": "#9467bd"}
    sns.barplot(data=df, x="cfg", y="pct", hue="resource", palette=pal,
                ax=ax1, width=0.62, gap=0.0, edgecolor="none")
    ymax1 = df["pct"].max()*1.28
    ax1.set_ylim(0, ymax1)
    for cont, r in zip(ax1.containers, res):
        for p, k in zip(cont, (0, 6)):
            h = p.get_height()
            ax1.text(p.get_x()+p.get_width()/2, h+ymax1*0.012, f"{END[k][r]:,}",
                     ha="center", va="bottom", fontsize=8.2, weight="bold",
                     color=pal[r])
    ax1.set_xlabel(""); ax1.set_ylabel("% of VC1902 PL resource")
    ax1.legend(title="", loc="upper right", fontsize=9.5)
    ax1.set_title("PL fabric: all-PL vs all-AIE (full model, measured)",
                  fontsize=11, weight="bold", pad=8)
    # reduction arrows
    for r in res:
        red = END[0][r]/END[6][r]
        ax1.text(0.5, 100*END[6][r]/PL_AVAIL[r] + ymax1*0.0,
                 "", ha="center")
    ax1.text(0.5, ymax1*0.93,
             f"DSP  {END[0]['DSP']}→{END[6]['DSP']}  ({END[0]['DSP']/END[6]['DSP']:.1f}×)\n"
             f"LUT  {END[0]['LUT']/1000:.0f}k→{END[6]['LUT']/1000:.0f}k  ({END[0]['LUT']/END[6]['LUT']:.1f}×)",
             ha="center", va="top", fontsize=9, color=GOOD, weight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GOOD, lw=0.7))

    # ---- RIGHT: tiles up + projected PL-DSP down vs blocks --------------
    k = np.array(BLOCKS_K)
    ax2.plot(k, AIE_TILES, "-o", color=AIE_COLOR, lw=2.3, ms=7,
             label="AIE tiles (measured)")
    ax2.set_xlabel("attention blocks on AIE  (of 6)")
    ax2.set_ylabel("AIE compute tiles", color=AIE_COLOR)
    ax2.tick_params(axis="y", labelcolor=AIE_COLOR)
    ax2.set_xticks(np.arange(0, 7)); ax2.set_ylim(0, 90)

    ax3 = ax2.twinx(); ax3.grid(False); ax3.spines["top"].set_visible(False)
    # projected PL DSP linear between measured endpoints 1055 (k=0) -> 314 (k=6)
    kf = np.arange(0, 7)
    dsp_proj = END[0]["DSP"] + (END[6]["DSP"]-END[0]["DSP"])/6.0 * kf
    ax3.plot(kf, dsp_proj, "--", color=PL_COLOR, lw=1.6, alpha=0.8,
             label="PL DSP (projected)")
    ax3.plot([0, 6], [END[0]["DSP"], END[6]["DSP"]], "s", color=PL_COLOR, ms=9,
             label="PL DSP (measured, full model)")
    ax3.set_ylabel("PL DSP slices", color=PL_COLOR)
    ax3.tick_params(axis="y", labelcolor=PL_COLOR)
    ax3.set_ylim(0, 1150)
    ax2.set_title("Scale AIE up → tiles up, PL DSP down",
                  fontsize=11, weight="bold", pad=8)
    l2_, lb2 = ax2.get_legend_handles_labels()
    l3_, lb3 = ax3.get_legend_handles_labels()
    ax2.legend(l2_+l3_, lb2+lb3, loc="center right", fontsize=8.6)

    fig.suptitle("Resource impact of scaling attention onto the AIE array",
                 y=1.02, fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_impact(path):
    """Full-model system impact: all-PL vs all-AIE — latency, throughput,
    power, accuracy; plus per-block AIE quantization error."""
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.3))
    cfgs = ["all-PL", "all-AIE"]
    colors = [PL_COLOR, AIE_COLOR]

    # latency
    ax = axes[0]
    vals = [END[0]["lat_ms"], END[6]["lat_ms"]]
    ax.bar(cfgs, vals, color=colors, width=0.6)
    _labels(ax, "{:.2f} ms", ymax=max(vals)*1.2)
    ax.set_ylim(0, max(vals)*1.25)
    ax.set_ylabel("per-event latency [ms]")
    ax.set_title("Latency", fontsize=11, weight="bold")
    ax.text(0.5, -0.20, f"{END[0]['lat_ms']/END[6]['lat_ms']:.2f}× faster",
            transform=ax.transAxes, ha="center", va="top",
            color=GOOD, weight="bold", fontsize=10)

    # throughput
    ax = axes[1]
    vals = [END[0]["thr"], END[6]["thr"]]
    ax.bar(cfgs, vals, color=colors, width=0.6)
    _labels(ax, "{:.0f} ev/s", ymax=max(vals)*1.2)
    ax.set_ylim(0, max(vals)*1.25)
    ax.set_ylabel("throughput [events/s]")
    ax.set_title("Throughput", fontsize=11, weight="bold")
    ax.text(0.5, -0.20, f"+{100*(END[6]['thr']/END[0]['thr']-1):.0f}%",
            transform=ax.transAxes, ha="center", va="top",
            color=GOOD, weight="bold", fontsize=10)

    # power
    ax = axes[2]
    vals = [END[0]["power_W"], END[6]["power_W"]]
    ax.bar(cfgs, vals, color=colors, width=0.6)
    _labels(ax, "{:.1f} W", ymax=max(vals)*1.2)
    ax.set_ylim(0, max(vals)*1.4)
    ax.set_ylabel("Versal device power [W]")
    ax.set_title("Power (measured, INA226)", fontsize=11, weight="bold")
    ax.text(0.5, -0.20, "~flat (static-dominated)",
            transform=ax.transAxes, ha="center", va="top",
            color=NEUTRAL, weight="bold", fontsize=9.5)

    # accuracy (per-block AIE int16 error + endpoint total)
    ax = axes[3]
    bl = list(PERBLOCK_ERR.keys())
    ev = [PERBLOCK_ERR[b] for b in bl]
    ax.bar(bl, ev, color=AIE_COLOR, width=0.62)
    _labels(ax, "{:.3f}", ymax=max(ev)*1.25, fs=9)
    ax.set_ylim(0, max(ev)*1.35)
    ax.set_ylabel("max |abs err| vs FP32 (per block)")
    ax.set_title("Accuracy cost of AIE int16", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelrotation=12)
    ax.text(0.5, -0.28,
            f"full model: MSE {END[0]['mse']:.3f}→{END[6]['mse']:.3f}  (+{END[6]['err_pct']:.0f}%)",
            transform=ax.transAxes, ha="center", va="top",
            color=BAD, weight="bold", fontsize=9)

    fig.suptitle("Full-model impact of offloading ALL attention to AIE  "
                 "(all-PL vs all-AIE, measured on VCK190)",
                 y=1.04, fontsize=12.5, weight="bold")
    fig.subplots_adjust(bottom=0.24)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_stage_breakdown(path):
    """Per-event critical-path contribution by pipeline stage.
    PL stages from the full-hybrid HLS csynth (cycles x 10 ns @100 MHz);
    AIE attention from measured HW round-trip (bridge_solo/cand_emu, x2 layers).
    NB: PL and AIE partially overlap in the per-event DATAFLOW, so bars show
    each stage's cost, not a strict sum; the point is the relative bottleneck."""
    # (label, microseconds, kind)
    stages = [
        ("pairwise MLP  (wij, 156 pairs serial)", 996.7, "PL"),
        ("obj  attn  x2 layers",                  422.0, "AIE"),
        ("cross attn  x2 layers  (est.)",         400.0, "AIE"),
        ("cand attn  x2 layers",                  135.0, "AIE"),
        ("embed",                                  47.2, "PL"),
        ("obj/cross PL bridges",                   41.0, "PL"),
        ("candidate_build x2",                     18.4, "PL"),
        ("misc PL (io/fork/remask)",               18.0, "PL"),
        ("ae_loss",                                13.5, "PL"),
        ("cand_lorentz",                            7.1, "PL"),
    ]
    stages.sort(key=lambda s: s[1])
    labels = [s[0] for s in stages]
    vals   = [s[1] for s in stages]
    cols   = [PL_COLOR if s[2] == "PL" else AIE_COLOR for s in stages]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("per-event time contribution  [µs]  (log)")
    ax.set_xscale("log"); ax.set_xlim(3, 2000)
    for yi, v in zip(y, vals):
        ax.text(v*1.05, yi, f"{v:.0f} µs", va="center", ha="left",
                fontsize=8.5, weight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PL_COLOR, label="PL fabric stage"),
                       Patch(color=AIE_COLOR, label="AIE attention (measured)")],
              loc="lower right", fontsize=9)
    ax.set_title("Per-event pipeline breakdown — pairwise MLP is the wall (~1.0 ms, 55%)",
                 fontsize=12, weight="bold", pad=10)
    ax.text(0.02, 0.04,
            "pipelining ceiling today = 1 / slowest stage = 1/pairwise ≈ 1,000 ev/s (1.8×)\n"
            "fix: unroll/pipeline the 156-pair MLP loop (currently serial) → ~0.1 ms\n"
            "then bottleneck shifts to the attention chain → AIE-tile scaling pays off",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.6,
            family="monospace", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=NEUTRAL, lw=0.7))
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


def fig_parallel(path):
    """Event-level parallelism: replicate to process events in parallel.
    Left  — measured throughput: isolated attention blocks vs full model.
    Mid   — per-replica resource ceiling: what binds, and at how many replicas.
    Right — aggregate throughput vs replicas, with the device ceiling marked."""
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- LEFT: measured throughput (log scale) -------------------------
    names = list(THR_BLOCK.keys()) + ["full model\n(6 attn + PL)"]
    vals  = list(THR_BLOCK.values()) + [THR_FULL]
    cols  = [AIE_COLOR, AIE_COLOR, PL_COLOR]
    ax0.bar(names, vals, color=cols, width=0.62)
    ax0.set_yscale("log")
    ax0.set_ylim(100, 30000)
    for i, v in enumerate(vals):
        ax0.text(i, v*1.08, f"{v:,}", ha="center", va="bottom",
                 fontsize=9.5, weight="bold")
    ax0.set_ylabel("throughput [events/s]  (log)")
    ax0.set_title("Measured throughput (HW, golden-fed)\nattention block >> full model",
                  fontsize=10.5, weight="bold")

    # --- MIDDLE: replica ceiling by binding resource -------------------
    cap = {"AIE tiles": AIE_AVAIL, "PL DSP": PL_AVAIL["DSP"], "PL LUT": PL_AVAIL["LUT"]}
    res_keys = ["AIE tiles", "PL DSP", "PL LUT"]
    res_attr = {"AIE tiles": "tiles", "PL DSP": "dsp", "PL LUT": "lut"}
    rep_names = list(REPLICA.keys())
    YCAP = 35.0   # axis limit; ceilings above this are drawn clipped + labelled
    x = np.arange(len(res_keys)); w = 0.36
    for j, rn in enumerate(rep_names):
        ceils = []
        for rk in res_keys:
            per = REPLICA[rn][res_attr[rk]]
            ceils.append(cap[rk] // per if per > 0 else np.inf)
        # clamp bar display height so off-scale ceilings don't blow up the canvas
        disp = [min(c, YCAP-1) if np.isfinite(c) else (YCAP-1) for c in ceils]
        color = PL_COLOR if "full" in rn else AIE_COLOR
        ax1.bar(x + (j-0.5)*w, disp, width=w, color=color,
                alpha=0.55 if "full" in rn else 0.9,
                label=rn.replace("\n", " "))
        for xi, c, d in zip(x + (j-0.5)*w, ceils, disp):
            txt = "∞" if not np.isfinite(c) else f"{int(c)}"
            ax1.text(xi, min(d, YCAP-1)+0.6, txt, ha="center",
                     va="bottom", fontsize=9, weight="bold", color=color)
    ax1.set_xticks(x); ax1.set_xticklabels(res_keys)
    ax1.set_ylabel("max parallel replicas on VC1902")
    ax1.set_ylim(0, YCAP)
    ax1.text(0.99, 0.97, "(bars clipped at 35; true value labelled)",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=7.5, color=NEUTRAL, style="italic")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.set_title("Event-parallel ceiling = min over resources\n"
                  "full model: PL/AIE bind @5–6   •   attn block: AIE-tile bind @30",
                  fontsize=10.5, weight="bold")

    # --- RIGHT: aggregate throughput vs replicas -----------------------
    R = np.arange(1, 33)
    obj_thr = THR_BLOCK["obj\nself-attn"]
    # attention-block replication (obj), AIE-tile bound at 30
    attn_ceiL = 400 // 13
    ya = np.where(R <= attn_ceiL, R*obj_thr, np.nan)
    ax2.plot(R, ya, "-o", ms=3, color=AIE_COLOR,
             label=f"obj attn block ×R  (bind {attn_ceiL})")
    ax2.axvline(attn_ceiL, color=AIE_COLOR, ls=":", lw=1.0)
    # full-model replication, AIE bound at 5
    full_ceiL = min(400//78, 1968//314)
    yf = np.where(R <= full_ceiL, R*THR_FULL, np.nan)
    ax2.plot(R, yf, "-s", ms=4, color=PL_COLOR,
             label=f"full model ×R  (bind {full_ceiL})")
    ax2.axvline(full_ceiL, color=PL_COLOR, ls=":", lw=1.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("parallel replicas R")
    ax2.set_ylabel("aggregate throughput [events/s]  (log)")
    ax2.set_title("Scale-out throughput\n(latency unchanged; data-parallel)",
                  fontsize=10.5, weight="bold")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.annotate(f"{attn_ceiL}×obj ≈ {attn_ceiL*obj_thr//1000}k ev/s",
                 (attn_ceiL, attn_ceiL*obj_thr),
                 textcoords="offset points", xytext=(-10, 8), ha="right",
                 fontsize=8.5, weight="bold", color=AIE_COLOR)

    fig.suptitle("Event-level parallelism: attention is AIE-tile-bound (~30×), "
                 "the full model is PL/AIE-bound (~5×)",
                 y=1.03, fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figs")
    os.makedirs(out, exist_ok=True)
    print("output dir:", out)
    fig_tiles    (os.path.join(out, "aie_scaling_tiles"))
    fig_resources(os.path.join(out, "aie_scaling_resources"))
    fig_impact   (os.path.join(out, "aie_scaling_impact"))
    fig_stage_breakdown(os.path.join(out, "aie_stage_breakdown"))
    fig_parallel (os.path.join(out, "aie_event_parallel"))
    print("Done.")
