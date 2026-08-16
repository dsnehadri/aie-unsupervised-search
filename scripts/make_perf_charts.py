#!/usr/bin/env python3
"""Generate publication-quality PL vs AIE comparison charts.

Data sources (already extracted from bench reports and aiesim timestamps):
- PL csynth .rpt under bench_attn_{obj,cand,cross}_proj/solution1/syn/report/
- AIE first-output latencies from PLIO timestamps in aiesimulator log
- Accuracy from check_attn_outputs.py runs (tol=0.1 pass on both)

Outputs (figs/):
- accuracy_per_block      — max abs err PL vs AIE vs FP32 ref
- latency_per_block       — end-to-end latency PL HLS vs AIE aiesim
- resources_per_block     — PL LUT/DSP/BRAM stacked vs AIE tile count
- compute_density         — effective MACs / us (PL beats AIE at this granularity)
- summary_table           — single tabular figure with all numbers
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

# ----------------------------------------------------------------------------
# Style — applied globally so every figure picks up the same look as
# fig_aie_vs_pl_steady (seaborn paper theme + Montserrat).
# ----------------------------------------------------------------------------
sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)
import matplotlib.font_manager as _fm
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

PL_COLOR    = "#d62728"   # blue
AIE_COLOR   = "#1f77b4"   # red
NEUTRAL     = "#7f7f7f"
RATIO_GOOD  = "#1a8a2e"
RATIO_BAD   = "#a31515"
PALETTE2    = {"AIE": AIE_COLOR, "PL": PL_COLOR}

BAR_WIDTH = 0.55  # matches fig_aie_vs_pl_steady


def _value_labels(ax, fmt, ymax=None, fontsize=9.5):
    """Bold value labels above every bar — matches steady chart."""
    if ymax is None:
        ymax = ax.get_ylim()[1]
    pad = ymax * 0.012
    for container in ax.containers:
        for patch in container:
            h = patch.get_height()
            if not np.isfinite(h) or h == 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            ax.text(x, h + pad, fmt.format(h),
                    ha="center", va="bottom",
                    fontsize=fontsize, weight="bold")


def _ratio_below(ax, xs, ratios, ymax=None, fontsize=9.5):
    """Under-tick per-group ratio annotation, green=better, red=worse.
    ratios is list of (text, "good"|"bad"|None) aligned with xs."""
    if ymax is None:
        ymax = ax.get_ylim()[1]
    for x, item in zip(xs, ratios):
        if item is None:
            continue
        txt, kind = item
        color = RATIO_GOOD if kind == "good" else (RATIO_BAD if kind == "bad" else NEUTRAL)
        ax.text(x, -ymax * 0.16, txt, ha="center", va="top",
                fontsize=fontsize, color=color, weight="bold")

BLOCKS = ["obj", "cand", "cross"]
LABELS = {
    "obj":   "Object\n(self-attn)",
    "cand":  "Candidate\n(self-attn)",
    "cross": "Cross\n(attn)",
}

# ----------------------------------------------------------------------------
# Data (extracted from reports / sim logs)
# ----------------------------------------------------------------------------
TOL = 0.1
ACC = {
    "obj":   {"PL": 0.029297, "AIE": 0.027615},
    "cand":  {"PL": 0.013672, "AIE": 0.010963},
    "cross": {"PL": 0.060547, "AIE": 0.056194},
}

# PL HLS csynth — bench_attn_{block}_proj/solution1/syn/report/attn_block_{block}_top_csynth.rpt
# target_clock = 5 ns (200 MHz); latency reported in cycles
PL = {
    "obj":   {"cycles": 20371, "LUT": 100217, "FF": 17049, "DSP": 146, "BRAM": 34, "URAM": 0,
              "est_period_ns": 3.605},
    "cand":  {"cycles":  4682, "LUT":  96571, "FF": 16764, "DSP": 137, "BRAM":  0, "URAM": 0,
              "est_period_ns": 3.624},
    "cross": {"cycles": 15167, "LUT":  97565, "FF": 16453, "DSP": 169, "BRAM": 20, "URAM": 0,
              "est_period_ns": 3.589},
}
PL_CLOCK_NS = 5.0   # target period

# AIE hw aiesim — first-output latency (PLIO start to last output sample)
AIE_LAT_US = {"obj": 496.944, "cand": 56.288, "cross": 235.520}

# AIE tile budget — per subgraph: 4 head_pre + 4 head_post + 5 post = 13 tiles
AIE_TILES = {"obj": 13, "cand": 13, "cross": 13}
AIE_TOTAL_TILES = 39
AIE_AVAILABLE_TILES = 400    # VC1902 AIE-1 has 400 compute tiles
AIE_CORE_FREQ_GHZ = 1.25

# VC1902 available resources (xcvc1902-vsva2197-2MP-e-S)
PL_AVAIL = {"LUT": 899840, "FF": 1799680, "DSP": 1968, "BRAM": 1934, "URAM": 463}

# ---------------------------------------------------------------------------
# Measured AIE per-event timing (from check_attn_outputs.py --timing-out)
# Loaded lazily so the script still works without the JSON.
# ---------------------------------------------------------------------------
AIE_TIMING_JSON = "/tmp/aie_timing.json"
def load_aie_timing():
    if not os.path.isfile(AIE_TIMING_JSON):
        return None
    import json
    with open(AIE_TIMING_JSON) as f:
        raw = json.load(f)
    out = {}
    for b in BLOCKS:
        first = np.array(raw[b]["first_ns"], dtype=np.float64) / 1000.0   # ns -> us
        last  = np.array(raw[b]["last_ns"],  dtype=np.float64) / 1000.0
        if first.size == 0:
            continue
        intervals = np.diff(first) if first.size > 1 else np.array([np.nan])
        out[b] = {
            "first_us":      first,             # per-event first-output time
            "last_us":       last,              # per-event last-output time
            "intervals_us":  intervals,         # inter-event spacing (steady state)
            "n_events":      int(first.size),
            "steady_us":     float(np.median(intervals)) if intervals.size and np.all(np.isfinite(intervals)) else np.nan,
            "fill_us":       float(first[0]),
        }
    return out
AIE_MEASURED = load_aie_timing()

# ----------------------------------------------------------------------------
# Derived
# ----------------------------------------------------------------------------
def pl_latency_us(b):
    return PL[b]["cycles"] * PL_CLOCK_NS / 1000.0

# Approximate MAC count per attention block (obj/cross use full N_MAX=12, cand uses T_DIM=3)
# obj/cross: QKV (3 * N * E * E) + score (H * N * Nkv * D) + ctx (H * N * Nkv * D) + Wo (N * E * E) + 3*FFN (N*E*E)
N_MAX, E_DIM, N_HEADS, D_HEAD, N_KV = 12, 16, 4, 4, 13
T_DIM, T_KV = 3, 4   # cand
def attn_macs(N, Nkv):
    qkv  = 3 * N * E_DIM * E_DIM
    sc   = N_HEADS * N * Nkv * D_HEAD
    ctx  = N_HEADS * N * Nkv * D_HEAD
    Wo   = N * E_DIM * E_DIM
    ffn  = 3 * N * E_DIM * E_DIM
    return qkv + sc + ctx + Wo + ffn
MACS = {"obj": attn_macs(N_MAX, N_KV), "cand": attn_macs(T_DIM, T_KV), "cross": attn_macs(N_MAX, N_KV)}

# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def fig_accuracy(path):
    rows = []
    for b in BLOCKS:
        label = LABELS[b].replace("\n", " ")
        rows.append({"block": label, "impl": "AIE", "err": ACC[b]["AIE"]})
        rows.append({"block": label, "impl": "PL",  "err": ACC[b]["PL"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=df, x="block", y="err", hue="impl",
                palette=PALETTE2, ax=ax, edgecolor="none",
                width=BAR_WIDTH, gap=0.0)

    ymax = max(df["err"].max(), TOL) * 1.20
    ax.set_ylim(0, ymax)
    ax.axhline(TOL, color=NEUTRAL, linestyle="--", linewidth=1.2)
    ax.text(-0.45, TOL * 1.04, f"tol = {TOL}",
            color=NEUTRAL, fontsize=9.5, weight="bold",
            va="bottom", ha="left")

    _value_labels(ax, "{:.3f}", ymax=ymax, fontsize=9.5)

    ax.set_xlabel("")
    ax.set_ylabel("max |abs error| vs FP32 reference")
    ax.legend(title="", loc="upper right", fontsize=10)
    ax.set_title("Numerical accuracy vs FP32 golden", fontsize=12, weight="bold", pad=10)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")

def fig_latency(path):
    """First-output latency (PL = cycles × 5 ns target; AIE = pipeline fill).

    This is the worst-case single-event latency. For the throughput-fair
    comparison see fig_aie_vs_pl_steady.
    """
    rows = []; n_by = {}
    for b in BLOCKS:
        pl_lat = pl_latency_us(b)
        meas = (AIE_MEASURED or {}).get(b)
        aie_fill = meas["fill_us"] if meas else AIE_LAT_US[b]
        n_by[b] = meas["n_events"] if meas else 0
        label = LABELS[b].replace("\n", " ") + f"\nN = {n_by[b]}"
        rows.append({"block": label, "impl": "AIE", "lat_us": aie_fill})
        rows.append({"block": label, "impl": "PL",  "lat_us": pl_lat})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=df, x="block", y="lat_us", hue="impl",
                palette=PALETTE2, ax=ax, edgecolor="none",
                width=BAR_WIDTH, gap=0.0)

    ymax = df["lat_us"].max() * 1.18
    ax.set_ylim(0, ymax)
    _value_labels(ax, "{:.1f} μs", ymax=ymax)

    ratios = []
    for b in BLOCKS:
        pl_lat = pl_latency_us(b)
        aie = (AIE_MEASURED or {}).get(b)
        aie_fill = aie["fill_us"] if aie else AIE_LAT_US[b]
        if aie_fill < pl_lat:
            ratios.append((f"{pl_lat/aie_fill:.2f}× faster", "good"))
        else:
            ratios.append((f"{aie_fill/pl_lat:.2f}× slower", "bad"))
    _ratio_below(ax, range(len(BLOCKS)), ratios, ymax=ymax)

    ax.set_xlabel("")
    ax.set_ylabel("first-output latency  [μs]")
    ax.legend(title="", loc="upper right", fontsize=10)
    ax.set_title("First-output latency: PL HLS csynth vs AIE pipeline fill",
                 fontsize=12, weight="bold", pad=10)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")

def fig_resources(path):
    """Side-by-side: PL utilization (% of VC1902) and AIE tile count.

    Restyled to match aie_vs_pl_steady: seaborn paper theme, bold value
    labels, consistent legend placement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # ---- PL utilization (% of VC1902 resources) -----------------------
    res_keys = ["LUT", "DSP", "BRAM"]
    res_palette = {"LUT": "#3a6fb0", "DSP": "#9467bd", "BRAM": "#17becf"}
    rows = []
    for b in BLOCKS:
        for k in res_keys:
            rows.append({"block": LABELS[b].replace("\n", " "),
                         "resource": k,
                         "pct": 100.0 * PL[b][k] / PL_AVAIL[k],
                         "raw": PL[b][k]})
    df_pl = pd.DataFrame(rows)
    sns.barplot(data=df_pl, x="block", y="pct", hue="resource",
                palette=res_palette, ax=ax1, edgecolor="none",
                width=BAR_WIDTH, gap=0.0)

    ymax1 = df_pl["pct"].max() * 1.30
    ax1.set_ylim(0, ymax1)
    # custom labels — raw counts, not pct
    rounds = {k: i for i, k in enumerate(res_keys)}
    for container, key in zip(ax1.containers, res_keys):
        for patch, b in zip(container, BLOCKS):
            raw = PL[b][key]
            h = patch.get_height()
            if not np.isfinite(h) or h <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            ax1.text(x, h + ymax1 * 0.012, f"{raw:,}",
                     ha="center", va="bottom",
                     fontsize=8.0, weight="bold", color=res_palette[key])
    ax1.set_xlabel("")
    ax1.set_ylabel("% of VC1902 PL resources")
    ax1.legend(title="", loc="upper right", fontsize=9.5)
    ax1.set_title("PL utilization per block", fontsize=11.5, weight="bold", pad=8)

    # ---- AIE tile count ----------------------------------------------
    rows = [{"block": LABELS[b].replace("\n", " "), "tiles": AIE_TILES[b]}
            for b in BLOCKS]
    df_aie = pd.DataFrame(rows)
    sns.barplot(data=df_aie, x="block", y="tiles",
                color=AIE_COLOR, ax=ax2, edgecolor="none",
                width=BAR_WIDTH, gap=0.0)
    ymax2 = max(t["tiles"] for t in rows) * 1.30
    ax2.set_ylim(0, ymax2)
    _value_labels(ax2, "{:.0f}", ymax=ymax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("AIE compute tiles")
    ax2.set_title(f"AIE tile usage  "
                  f"({AIE_TOTAL_TILES}/{AIE_AVAILABLE_TILES} = "
                  f"{100*AIE_TOTAL_TILES/AIE_AVAILABLE_TILES:.1f}% of array)",
                  fontsize=11.5, weight="bold", pad=8)

    fig.suptitle("Compute resource cost: PL vs AIE", y=1.02,
                 fontsize=12.5, weight="bold")
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")

def fig_compute_density(path):
    """Effective MAC throughput per block (steady state).

    PL: MACs / per-event latency (no pipelining across events).
    AIE: MACs / steady-state inter-event interval (after pipeline fill).
    """
    rows = []
    ratios = []
    for b in BLOCKS:
        pl_lat = pl_latency_us(b)
        pl_g = MACS[b] / (pl_lat * 1e-6) / 1e9
        meas = (AIE_MEASURED or {}).get(b)
        if meas and np.isfinite(meas["steady_us"]):
            aie_lat = meas["steady_us"]; n = meas["n_events"]
        else:
            aie_lat = AIE_LAT_US[b]; n = 0
        aie_g = MACS[b] / (aie_lat * 1e-6) / 1e9
        label = LABELS[b].replace("\n", " ") + f"\nN = {n}"
        rows.append({"block": label, "impl": "AIE", "gops": aie_g})
        rows.append({"block": label, "impl": "PL",  "gops": pl_g})
        # ratio: higher gops is better, so "good" if AIE > PL
        if aie_g > pl_g:
            ratios.append((f"{aie_g/pl_g:.2f}× faster", "good"))
        else:
            ratios.append((f"{pl_g/aie_g:.2f}× slower", "bad"))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=df, x="block", y="gops", hue="impl",
                palette=PALETTE2, ax=ax, edgecolor="none",
                width=BAR_WIDTH, gap=0.0)
    ymax = df["gops"].max() * 1.18
    ax.set_ylim(0, ymax)
    _value_labels(ax, "{:.2f}", ymax=ymax)
    _ratio_below(ax, range(len(BLOCKS)), ratios, ymax=ymax)

    ax.set_xlabel("")
    ax.set_ylabel("effective throughput  [Gop/s]")
    ax.legend(title="", loc="upper right", fontsize=10)
    ax.set_title("Steady-state MAC throughput per attention block",
                 fontsize=12, weight="bold", pad=10)

    # MACs reference, bottom-left
    macs_txt = "MAC count assumed:  " + "  •  ".join(
        f"{b}: {MACS[b]/1000:.1f} K" for b in BLOCKS)
    fig.text(0.02, 0.01, macs_txt, fontsize=8, color=NEUTRAL,
             family="monospace", ha="left", va="bottom")
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")

def fig_throughput_crossover(path):
    """Wall-clock time vs batch size, per block.

    PL: every event takes pl_latency (no pipelining across events).
    AIE: measured dots from aiesim PLIO timestamps; dashed extrapolation
    uses the steady-state inter-event interval observed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True)
    N_full = np.arange(1, 101)

    for ax, b in zip(axes, BLOCKS):
        pl_lat = pl_latency_us(b)
        pl_y = N_full * pl_lat
        ax.plot(N_full, pl_y, color=PL_COLOR, linewidth=2.0,
                label=f"PL  ({pl_lat:.0f} μs/evt)")

        meas = AIE_MEASURED.get(b) if AIE_MEASURED else None
        if meas and meas["n_events"] >= 1:
            n_meas = meas["n_events"]
            x_meas = np.arange(1, n_meas + 1)
            y_meas = meas["last_us"]
            steady = meas["steady_us"]
            fill = meas["fill_us"]

            ax.plot(x_meas, y_meas, color=AIE_COLOR, marker="o", markersize=2.5,
                    linewidth=1.6, label=f"AIE measured (N={n_meas})")
            if n_meas < N_full.size and np.isfinite(steady):
                N_extra = np.arange(n_meas + 1, N_full.size + 1)
                y_extra = y_meas[-1] + (N_extra - n_meas) * steady
                ax.plot(N_extra, y_extra, color=AIE_COLOR, linestyle="--",
                        linewidth=1.4, alpha=0.8,
                        label=f"AIE extrapolated (Δ={steady:.1f}μs/evt)")

            # crossover (if AIE eventually beats PL)
            denom = pl_lat - steady
            if denom > 0:
                n_star = (fill - steady) / denom
                if 1 < n_star < N_full.max() * 1.5:
                    ax.axvline(n_star, color=NEUTRAL, linestyle=":", linewidth=1.0)
                    ax.text(n_star + 1, fill * 0.30, f"crossover\nN ≈ {n_star:.0f}",
                            color=NEUTRAL, fontsize=8.5)
                badge = (f"{pl_lat/steady:.2f}× faster (steady)", "good")
            else:
                ax.text(0.98, 0.03,
                        "Δ_ss > PL μs/evt:\nAIE never wins\nat any batch size",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color=RATIO_BAD, weight="bold")
                badge = (f"{steady/pl_lat:.2f}× slower (steady)", "bad")

            # top-left stats box
            ax.text(0.02, 0.98,
                    f"fill = {fill:.0f} μs\n"
                    f"Δ_ss = {steady:.1f} μs/evt\n"
                    f"PL    = {pl_lat:.0f} μs/evt",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.5, color="#333", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=NEUTRAL, linewidth=0.6))
        else:
            ax.text(0.5, 0.5, "no AIE timing data\n(/tmp/aie_timing.json)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=NEUTRAL)
            badge = None

        ax.set_title(LABELS[b].replace("\n", " "), fontsize=11.5, weight="bold")
        ax.set_xlabel("batch size  (events)")
        if b == BLOCKS[0]:
            ax.set_ylabel("wall-clock time  [μs]")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(1, N_full.max())

        if badge is not None:
            txt, kind = badge
            color = RATIO_GOOD if kind == "good" else RATIO_BAD
            ax.text(0.5, -0.18, txt, transform=ax.transAxes, ha="center",
                    va="top", fontsize=10, color=color, weight="bold")

    fig.suptitle("Latency vs throughput crossover — PL stays linear, "
                 "AIE amortizes pipeline fill",
                 y=1.02, fontsize=12.5, weight="bold")
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")


def fig_aie_vs_pl_steady(path):
    """All 3 blocks side-by-side: AIE steady-state vs PL per-event latency."""
    rows = []
    ratios = []
    for b in BLOCKS:
        pl_lat = pl_latency_us(b)
        meas   = (AIE_MEASURED or {}).get(b)
        if meas and meas["n_events"] >= 2 and np.isfinite(meas["steady_us"]):
            aie_val = meas["steady_us"]
            n_int   = meas["intervals_us"].size
        else:
            aie_val = float("nan"); n_int = 0
        block_label = LABELS[b].replace("\n", " ") + f"\nN = {n_int}"
        rows.append({"block": block_label, "impl": "AIE", "latency_us": aie_val})
        rows.append({"block": block_label, "impl": "PL",  "latency_us": pl_lat})
        if np.isfinite(aie_val):
            if aie_val < pl_lat:
                ratios.append((f"{pl_lat/aie_val:.2f}× faster", "good"))
            else:
                ratios.append((f"{aie_val/pl_lat:.2f}× slower", "bad"))
        else:
            ratios.append(None)
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=df, x="block", y="latency_us", hue="impl",
                palette=PALETTE2, ax=ax, edgecolor="none",
                width=BAR_WIDTH, gap=0.0)
    ymax = df["latency_us"].max() * 1.18
    ax.set_ylim(0, ymax)
    _value_labels(ax, "{:.1f} μs", ymax=ymax)
    _ratio_below(ax, range(len(BLOCKS)), ratios, ymax=ymax)

    ax.set_xlabel("")
    ax.set_ylabel("steady-state per-event latency  [μs]")
    ax.legend(title="", loc="upper right", fontsize=10)
    ax.set_title("AIE steady-state vs PL per-event latency",
                 fontsize=12, weight="bold", pad=10)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")


def fig_per_event_latency(path):
    """Per-event AIE latency vs event index, with PL reference line.

    Event 1 = pipeline fill (gray bar); events 2..N = inter-event interval
    (red bars). PL reference is a constant blue line.
    """
    if not AIE_MEASURED:
        print("skip per_event_latency: no AIE timing JSON")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    for ax, b in zip(axes, BLOCKS):
        meas = AIE_MEASURED.get(b)
        if not meas or meas["n_events"] < 1:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        pl_lat = pl_latency_us(b)
        fill = meas["fill_us"]
        steady = meas["steady_us"]
        ints = meas["intervals_us"]
        n_meas = meas["n_events"]

        per_evt = np.concatenate([[fill], ints]) if ints.size else np.array([fill])
        x = np.arange(1, per_evt.size + 1)

        # bar colors: gray fill bar then red steady bars
        colors = [NEUTRAL] + [AIE_COLOR] * (per_evt.size - 1)
        ax.bar(x, per_evt, color=colors, width=0.9, edgecolor="none")

        ax.axhline(pl_lat, color=PL_COLOR, linestyle="-", linewidth=1.8,
                   label=f"PL  {pl_lat:.0f} μs/evt")
        if np.isfinite(steady):
            ax.axhline(steady, color=AIE_COLOR, linestyle="--", linewidth=1.2,
                       alpha=0.75, label=f"AIE Δ_ss median  {steady:.1f} μs")

        ax.annotate(f"pipeline\nfill: {fill:.0f} μs",
                    xy=(1, fill), xytext=(min(n_meas * 0.18, 6), fill * 0.78),
                    fontsize=9, color=NEUTRAL, weight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.7))

        if np.isfinite(steady):
            if steady < pl_lat:
                ratio_txt = f"{pl_lat/steady:.2f}× faster"
                ratio_color = RATIO_GOOD
            else:
                ratio_txt = f"{steady/pl_lat:.2f}× slower"
                ratio_color = RATIO_BAD
            ax.text(0.5, -0.18, ratio_txt, transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=10, color=ratio_color, weight="bold")

        ax.set_title(LABELS[b].replace("\n", " "), fontsize=11.5, weight="bold")
        ax.set_xlabel("event index")
        if b == BLOCKS[0]:
            ax.set_ylabel("per-event latency  [μs]")
        ax.legend(loc="center right", fontsize=8.5, framealpha=0.9, frameon=True)
        ax.set_xlim(0.3, per_evt.size + 0.7)
        ax.set_ylim(0, max(per_evt.max(), pl_lat) * 1.20)

    fig.suptitle("AIE per-event latency (aiesim, hw target).  "
                 "Event 1 = pipeline fill, events 2..N = inter-event interval.",
                 y=1.02, fontsize=12.5, weight="bold")
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")


def fig_block_diagram(path):
    """Vertical block diagram of the pipeline, showing AIE vs PL forks.

    PL boxes blue, AIE boxes red; arrows in gray. Inherits Montserrat from
    the global rcParams.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    PL_BLUE = PL_COLOR
    AIE_RED = AIE_COLOR
    ARROW   = "#444444"

    fig, ax = plt.subplots(figsize=(2.89, 4.43))
    ax.set_xlim(0, 10); ax.set_ylim(0, 22)
    ax.axis("off")

    XC = 5.0
    XA = 2.55
    XP = 7.45
    BOX_W = 3.9
    BOX_H = 1.7  # two lines of text
    FS = 5.6     # box font size

    Y = {
        "embed":      20.7,
        "obj":        17.8,
        "build_a":    14.9,
        "cand":       12.0,
        "cross":       9.1,
        "build_b":     6.2,
        "autoenc":     3.3,
    }

    def box(x, y, text, color):
        bx = FancyBboxPatch(
            (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
            boxstyle="round,pad=0.03,rounding_size=0.16",
            linewidth=0, facecolor=color, edgecolor="none",
        )
        ax.add_patch(bx)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=FS, color="white", weight="bold",
                linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ar = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=6,
            linewidth=0.8, color=ARROW,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(ar)

    box(XC, Y["embed"],   "Embedding\n(PL)",                  PL_BLUE)

    box(XA, Y["obj"],     "Object Self-Attention\n(AIE)",     AIE_RED)
    box(XP, Y["obj"],     "Object Self-Attention\n(PL)",      PL_BLUE)

    box(XC, Y["build_a"], "Build Candidates\n(PL)",           PL_BLUE)

    box(XA, Y["cand"],    "Candidate Self-Attention\n(AIE)",  AIE_RED)
    box(XP, Y["cand"],    "Candidate Self-Attention\n(PL)",   PL_BLUE)

    box(XA, Y["cross"],   "Cross-Attention\n(AIE)",           AIE_RED)
    box(XP, Y["cross"],   "Cross-Attention\n(PL)",            PL_BLUE)

    box(XC, Y["build_b"], "Build Candidates\n(PL)",           PL_BLUE)

    box(XC, Y["autoenc"], "Autoencoder\n(PL)",                PL_BLUE)

    # Arrows (top -> bottom)
    arrow(XC, Y["embed"]   - BOX_H/2, XA, Y["obj"]    + BOX_H/2)
    arrow(XC, Y["embed"]   - BOX_H/2, XP, Y["obj"]    + BOX_H/2)
    arrow(XA, Y["obj"]     - BOX_H/2, XC, Y["build_a"] + BOX_H/2)
    arrow(XP, Y["obj"]     - BOX_H/2, XC, Y["build_a"] + BOX_H/2)
    arrow(XC, Y["build_a"] - BOX_H/2, XA, Y["cand"]   + BOX_H/2)
    arrow(XC, Y["build_a"] - BOX_H/2, XP, Y["cand"]   + BOX_H/2)
    arrow(XA, Y["cand"]    - BOX_H/2, XA, Y["cross"]  + BOX_H/2)
    arrow(XP, Y["cand"]    - BOX_H/2, XP, Y["cross"]  + BOX_H/2)
    arrow(XA, Y["cross"]   - BOX_H/2, XC, Y["build_b"] + BOX_H/2)
    arrow(XP, Y["cross"]   - BOX_H/2, XC, Y["build_b"] + BOX_H/2)
    arrow(XC, Y["build_b"] - BOX_H/2, XC, Y["autoenc"] + BOX_H/2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    print(f"wrote {path}.{{pdf,png}}")


def fig_summary_table(path):
    rows = []
    sum_pl_lat = 0.0
    sum_aie_fill = 0.0
    sum_aie_ss   = 0.0
    for b in BLOCKS:
        pl_lat = pl_latency_us(b)
        meas   = (AIE_MEASURED or {}).get(b)
        if meas and meas["n_events"] >= 2 and np.isfinite(meas["steady_us"]):
            aie_fill = meas["fill_us"]
            aie_ss   = meas["steady_us"]
            n_int    = meas["intervals_us"].size
            ratio    = aie_ss / pl_lat
            ratio_str = (f"{pl_lat/aie_ss:.2f}× faster" if aie_ss < pl_lat
                         else f"{aie_ss/pl_lat:.2f}× slower")
            aie_fill_str = f"{aie_fill:.1f}"
            aie_ss_str   = f"{aie_ss:.1f}"
            n_str        = f"{n_int}"
        else:
            aie_fill = AIE_LAT_US[b]; aie_ss = float("nan")
            ratio    = float("nan"); ratio_str = "—"
            aie_fill_str = f"{aie_fill:.1f}"; aie_ss_str = "—"; n_str = "0"
        sum_pl_lat   += pl_lat
        sum_aie_fill += aie_fill
        if np.isfinite(aie_ss): sum_aie_ss += aie_ss
        rows.append([
            b,
            f"{ACC[b]['PL']:.4f}", f"{ACC[b]['AIE']:.4f}",
            f"{pl_lat:.1f}",
            aie_fill_str,
            aie_ss_str,
            ratio_str,
            n_str,
            f"{PL[b]['LUT']:,}",
            f"{PL[b]['DSP']}",
            f"{AIE_TILES[b]}",
        ])
    # totals row
    if sum_aie_ss > 0:
        tot_ratio = (f"{sum_pl_lat/sum_aie_ss:.2f}× faster" if sum_aie_ss < sum_pl_lat
                     else f"{sum_aie_ss/sum_pl_lat:.2f}× slower")
    else:
        tot_ratio = "—"
    rows.append([
        "Total", "—", "—",
        f"{sum_pl_lat:.1f}",
        f"{sum_aie_fill:.1f}",
        f"{sum_aie_ss:.1f}" if sum_aie_ss > 0 else "—",
        tot_ratio,
        "—",
        f"{sum(PL[b]['LUT'] for b in BLOCKS):,}",
        f"{sum(PL[b]['DSP'] for b in BLOCKS)}",
        f"{AIE_TOTAL_TILES}",
    ])
    headers = ["block", "PL err", "AIE err",
               "PL μs/evt", "AIE fill μs", "AIE Δ_ss μs",
               "AIE vs PL (steady)", "AIE N",
               "PL LUT", "PL DSP", "AIE tiles"]
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.55)
    n_cols = len(headers)
    ratio_col = headers.index("AIE vs PL (steady)")
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0"); cell.set_text_props(weight="bold")
        elif r == len(rows):
            cell.set_facecolor("#fafafa"); cell.set_text_props(weight="bold")
        # Color the ratio column
        if c == ratio_col and r > 0 and r <= len(rows):
            txt = cell.get_text().get_text()
            if "faster" in txt:
                cell.set_text_props(color="#1a8a2e", weight="bold")
            elif "slower" in txt:
                cell.set_text_props(color="#a31515", weight="bold")
    ax.set_title("PL vs AIE — attention block summary "
                 "(VC1902, PL @ 200 MHz / AIE @ 1.25 GHz)",
                 pad=10, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print(f"wrote {path}.{{pdf,png}}")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figs")
    os.makedirs(out, exist_ok=True)
    print(f"output dir: {out}")
    fig_accuracy        (os.path.join(out, "accuracy_per_block"))
    fig_latency         (os.path.join(out, "latency_per_block"))
    fig_resources       (os.path.join(out, "resources_per_block"))
    fig_compute_density   (os.path.join(out, "compute_density"))
    fig_throughput_crossover(os.path.join(out, "throughput_crossover"))
    fig_per_event_latency (os.path.join(out, "per_event_latency"))
    fig_aie_vs_pl_steady  (os.path.join(out, "aie_vs_pl_steady"))
    fig_block_diagram     (os.path.join(out, "block_diagram"))
    fig_summary_table     (os.path.join(out, "summary_table"))
    print("\nDone.")
