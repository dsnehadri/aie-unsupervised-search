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
    _bi = "/home/snehadri/aie_scratch_save_20260810/block_intervals.json"
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
        pl_us, pl_label = pl_iso, "PL block, isolated at 100 MHz"
    if aie_us is None:
        axb.bar(xs, pl_us, width=0.55, color=PL_C, label="PL block, in the all-PL design")
        for x, v in zip(xs, pl_us):
            axb.text(x, v + 4, f"{v:.0f}", ha="center", fontsize=10.5)
        axb.axhline(HYB_INTERVAL, color=AIE_C, lw=2, ls="--",
                    label="Hybrid interval: no AIE block exceeds this")
    else:
        w = 0.36
        axb.bar(xs - w/2, pl_us, width=w, color=PL_C, label=pl_label)
        axb.bar(xs + w/2, aie_us, width=w, color=AIE_C, label="AIE block, isolated, measured")
        for x, v in zip(xs - w/2, pl_us):
            axb.text(x, v + 4, f"{v:.0f}", ha="center", fontsize=10)
        for x, v in zip(xs + w/2, aie_us):
            axb.text(x, v + 4, f"{v:.0f}", ha="center", fontsize=10)
    axb.set_xticks(xs)
    axb.set_xticklabels(blocks, fontsize=11.5)
    axb.set_ylabel("Time per event [µs]", fontsize=12.5)
    axb.set_ylim(0, max(pl_us.max(), 210) * 1.22)
    axb.legend(fontsize=10, frameon=False, loc="upper right")

    # --- (b) AIE tile replication, with the shared-feeder model ---
    # One PL feeder dispatches events to the N instances in turn, so the time
    # per event is t = t_f + t_c/N: t_f = feeder time per event, t_c = the
    # per-invocation cost every instance pays (launch + block latency).
    # Throughput = 1/t = N/(t_c + N t_f) -> 1/t_f as N grows. Fit on the
    # invocation time T(N) = t_c + N t_f, which is linear in N.
    # Prefer the sweep on the CURRENT 12-tile graph (obj24 vehicle, cores kept
    # out of the gated column 0) when its CSV exists; fall back to the July
    # 13-tile obj16 sweep otherwise.
    import csv, os
    SAVE = "/home/snehadri/aie_scratch_save_20260810"
    TILES_PER = 12

    def _sweep(path):
        rows = [r for r in csv.DictReader(open(path))]
        tiles = np.array([float(r["tiles"]) for r in rows])
        meas = np.array([float(r["agg_ev_s"]) for r in rows])
        n = tiles / TILES_PER
        t_f, t_c = np.polyfit(n, n / meas, 1)   # T(N) = t_c + N t_f, per invocation
        return tiles, meas, t_f, t_c

    # Two kernel versions of the same 24-instance vehicle. The vector integer
    # layer norm cuts t_c (the per-instance compute) but leaves t_f (the shared
    # PL feeder) alone, so both curves run into the same ceiling -- the faster
    # kernels simply get there with fewer tiles. That vehicle stops configuring
    # reliably above 6 instances with the faster kernels, hence the shorter series.
    SERIES = [
        (f"{SAVE}/obj24_sweep_c0.csv",         "float layer norm",          AIE_C,   "o"),
        (f"{SAVE}/obj24_sweep_lnv2_clean.csv", "vector integer layer norm", "#2ca02c", "s"),
    ]
    top = 0
    for path, lab, col, mk in SERIES:
        if not os.path.isfile(path):
            continue
        tiles, meas, t_f, t_c = _sweep(path)
        n_model = np.linspace(0.6, 400 / TILES_PER, 400)
        thr_model = n_model / (t_c + n_model * t_f)
        top = max(top, thr_model.max())
        # solid where the vehicle was measured, dashed where the model extrapolates
        inside = n_model * TILES_PER <= tiles.max()
        axs.plot(n_model[inside] * TILES_PER, thr_model[inside], "-", color=col, lw=1.6, alpha=.85,
                 label=(f"{lab}: $t_f$ = {t_f*1e6:.1f} µs, $t_c$ = {t_c*1e6:.0f} µs"))
        axs.plot(n_model[~inside] * TILES_PER, thr_model[~inside], ":", color=col, lw=1.6, alpha=.85)
        axs.plot(tiles, meas, mk, color=col, ms=7, markeredgecolor="k",
                 markeredgewidth=0.4, zorder=5)
    axs.axhline(6334, color=PL_C, lw=2, ls="--",
                label="PL block, isolated at 100 MHz")
    _tk = [12, 48, 96, 192, 288, 400]
    axs.set_xticks(_tk)
    axs.set_xticklabels([f"{int(v)}" for v in _tk])
    axs.set_xlim(0, 420)
    axs.set_xlabel("AI Engine tiles", fontsize=12.5)
    axs.set_ylabel("Object attention block throughput [events / s]",
                   fontsize=12.5)
    axs.set_ylim(0, top * 1.15)
    axs.legend(fontsize=9.2, loc="upper left", bbox_to_anchor=(0.02, 0.94), frameon=False,
               title=r"Model $t = t_f + t_c/N$", title_fontsize=9.5)
    axs.get_legend().get_title().set_ha("left")

    save(fig, "throughput_blocks_and_scaling")


if __name__ == "__main__":
    fig_endtoend()
    fig_blocks_and_scaling()
