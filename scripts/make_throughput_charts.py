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
    rows = [  # (label, ev/s, color) -- all measured, 2000-event runs
        ("all-PL  baseline",                    478,  PL_C),
        ("all-PL  optimized kernels",          1139,  PL_C),
        ("all-PL  batched dataflow",           4869,  PL_C),
        ("all-PL  attention optimised",         50870, PL_C),
        ("all-PL  faster clock",               63780, PL_C),
        ("all-PL  current",                    86229, PL_C),
        ("AIE hybrid  baseline",                551,  AIE_C),
        ("AIE hybrid  pipelined bridge",       7549,  AIE_C),
        ("AIE hybrid  whole stack on array",  135000, AIE_C),
        ("AIE hybrid  91 tiles",              153106, AIE_C),
        ("AIE hybrid  embedding on array",    183190, AIE_C),
        ("AIE hybrid  current",               201326, AIE_C),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows], float)
    cols = [r[2] for r in rows]
    ys = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(ys, vals, color=cols, height=0.62)
    for y, v in zip(ys, vals):
        ax.text(v * 1.02, y, f"{v:,.0f}", va="center", fontsize=10.5, weight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("throughput  [events / s]")
    ax.set_xlim(0, vals.max() * 1.14)
    ax.set_title("End-to-end throughput on VCK190",
                 fontsize=12.5, pad=10)
    handles = [mpl.patches.Patch(color=PL_C, label="all-PL (AUC 0.9825)"),
               mpl.patches.Patch(color=AIE_C, label="AIE hybrid (AUC 0.9825)")]
    ax.legend(handles=handles, loc="upper right", fontsize=10,
              framealpha=0.95)
    save(fig, "throughput_endtoend")


# ---------------------------------------------------------------------------
# 2. Per-block single-instance throughput, PL vs AIE (runtime @100 MHz)
# ---------------------------------------------------------------------------
def fig_blocks_and_scaling():
    """Two panels: single-block PL vs AIE, and AIE tile replication."""
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    # --- (a) attention-block cost inside the DEPLOYED designs ---
    # Earlier isolated-vehicle bars were not comparable: the standalone PL
    # blocks link at 100 MHz, but the full PL design closes timing only at
    # 80 MHz (83% DSP), and the standalone AIE vehicle sends one event per
    # invocation so its number was dominated by ~200 us of launch overhead.
    # These are the per-event costs the blocks actually incur in the shipped
    # pipelines: PL from the routed design at its real 80 MHz clock, and the
    # hybrid's measured 111 us/event interval as the bound on every AIE block
    # (each one passes every event, so none can be slower than the pipeline).
    blocks = ["Object\nattention", "Candidate\nattention", "Cross\nattention"]
    pl_us = np.array([210.3, 44.4, 163.8], float)      # 80 MHz, routed cycles
    HYB_INTERVAL = 111.0
    xs = np.arange(len(blocks))
    import json, os
    _bi = "/home/snehadri/aie_scratch_save_20260810/block_intervals_v5.json"   # blocks3_v2 + pl_attn_v3
    aie_us = None
    if os.path.isfile(_bi):
        _d = json.load(open(_bi))
        keys = ["Object attention", "Candidate attention", "Cross attention"]
        if all(k in _d for k in keys):
            aie_us = np.array([_d[k]["slope_us"] for k in keys], float)
    pl_iso = None
    if os.path.isfile(_bi):
        _d = json.load(open(_bi))
        pk = ["PL Object attention", "PL Candidate attention", "PL Cross attention"]
        if all(k in _d for k in pk):
            pl_iso = np.array([_d[k]["slope_us"] for k in pk], float)
    pl_label = "PL block, in the all-PL design"
    if pl_iso is not None:            # like-for-like: both sides isolated, batch-measured
        pl_us, pl_label = pl_iso, "PL block"
    if aie_us is None:
        axb.bar(xs, pl_us, width=0.55, color=PL_C, label="PL block, in the all-PL design")
        for x, v in zip(xs, pl_us):
            axb.text(x, v + 4, f"{v:.0f}", ha="center", fontsize=10.5)
        axb.axhline(HYB_INTERVAL, color=AIE_C, lw=2, ls="--",
                    label="Hybrid interval: no AIE block exceeds this")
    else:
        w = 0.36
        axb.bar(xs - w/2, pl_us, width=w, color=PL_C, label=pl_label)
        axb.bar(xs + w/2, aie_us, width=w, color=AIE_C, label="AIE block")
        _off = max(pl_us.max(), aie_us.max()) * 0.015
        for x, v in zip(xs - w/2, pl_us):
            axb.text(x, v + _off, f"{v:.1f}", ha="center", fontsize=10)
        for x, v in zip(xs + w/2, aie_us):
            axb.text(x, v + _off, f"{v:.1f}", ha="center", fontsize=10)
    axb.set_xticks(xs)
    axb.set_xticklabels(blocks, fontsize=11.5)
    axb.set_ylabel("Time per event [µs]", fontsize=12.5)
    axb.set_ylim(0, pl_us.max() * 1.22)
    axb.legend(fontsize=10, frameon=False, loc="upper right")

    # --- (b) AIE tile replication ---
    # obj20_v5: 20 object blocks (v5 array flags, 15 tiles each) fed by one fabric
    # feeder that sends a preloaded event to every active instance in the same
    # clock cycle, many events per call. Each point is the slope of call time vs
    # events per call, so launch cost is excluded, as in the left panel.
    # (The older sweep sent one event per instance per call through a serial
    # feeder; its ~16 us/event feeder, not the array, capped the curve.)
    # One and two instances run at the single-block rate (4.3 us); from the third
    # instance every round takes 7.6 us: the feeder moves in lockstep, so the
    # slowest instance (the third) sets the pace for all.
    import os
    SAVE = "/home/snehadri/aie_scratch_save_20260810"
    TILES_PER = 15   # 4 pre, 4 head post, 2 merge, projection, b1, b2, c1, c
    tiles, agg = [], []
    for l in open(f"{SAVE}/obj20_sweep_v5_batch.csv"):
        if l.startswith("FIT,"):
            kv = dict(x.split("=") for x in l.strip().split(",")[1:])
            tiles.append(int(kv["tiles"])); agg.append(float(kv["agg_ev_s"]))
    tiles, agg = np.array(tiles), np.array(agg)
    axs.plot(tiles, agg / 1e6, "-o", color=AIE_C, lw=1.6, ms=7, markeredgecolor="k",
             markeredgewidth=0.4, zorder=5, label="AIE blocks")
    # one PL object block, from the same per-block sweep as the left panel
    axs.axhline(1 / pl_us[0], color=PL_C, lw=2, ls="--", label="PL block")
    _tk = [15, 60, 120, 180, 240, 300]
    axs.set_xticks(_tk)
    axs.set_xticklabels([f"{int(v)}" for v in _tk])
    axs.set_xlim(0, 320)
    axs.set_xlabel("AI Engine tiles", fontsize=12.5)
    axs.set_ylabel("Object attention throughput [million events / s]", fontsize=12.5)
    axs.set_ylim(0, agg.max() / 1e6 * 1.12)
    axs.legend(fontsize=10, loc="upper left", frameon=False)

    save(fig, "throughput_blocks_and_scaling")


if __name__ == "__main__":
    fig_endtoend()
    fig_blocks_and_scaling()
