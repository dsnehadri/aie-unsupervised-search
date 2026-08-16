#!/usr/bin/env python3
"""SCHEMATIC (not a measurement): why the big attention latency bar is not the
throughput bottleneck. The six attention blocks are separate pipeline stages;
in the cross-event-pipelined design they process different events at once, so
one event's latency = sum of blocks (~758 us of attention / ~0.92 ms total),
while a new event completes every pipeline interval (~0.13 ms -> 7,549 ev/s).

Block widths are drawn EQUAL for clarity (illustrative); the annotated latency
and throughput numbers are the measured ones.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_theme(context="paper", style="white", palette="deep", font_scale=1.1)
import matplotlib.font_manager as _fm
if any("ontserrat" in (f or "").lower() for f in _fm.findSystemFonts()):
    plt.rcParams["font.family"] = "Montserrat"
mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 220, "axes.titleweight": "bold"})
NEUTRAL = "#7f7f7f"

STAGES = ["obj0", "cand0", "cross0", "obj1", "cand1", "cross1"]   # 6 attention blocks
N_STAGES = len(STAGES)
W = 758.0 / N_STAGES        # illustrative block width so one event spans 758 us
N_EVENTS = 8                # events in flight
EVENT_COLORS = sns.color_palette("crest", N_EVENTS)


def fig(path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for e in range(N_EVENTS):
        for s in range(N_STAGES):
            x0 = e*W + s*W
            ax.barh(N_STAGES-1-s, W, left=x0, height=0.8,
                    color=EVENT_COLORS[e], edgecolor="white", linewidth=1.2)
    ax.set_yticks(range(N_STAGES))
    ax.set_yticklabels(STAGES[::-1], fontsize=11)
    ax.set_ylabel("attention block (pipeline stage)")
    ax.set_xlabel("time  [µs]")
    ax.set_xlim(0, (N_EVENTS + N_STAGES - 1) * W)
    ax.set_ylim(-1.6, N_STAGES + 1.4)

    # one event's latency span (event index 2's diagonal, first->last block)
    e_hi = 2
    x_start = e_hi*W
    x_end = e_hi*W + N_STAGES*W
    ax.annotate("", xy=(x_end, -0.75), xytext=(x_start, -0.75),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.8))
    ax.text((x_start+x_end)/2, -1.15,
            "latency: one event through all 6 blocks  ≈ 758 µs attention  (0.92 ms end-to-end, measured)",
            ha="center", va="center", fontsize=9.5, weight="bold", color="#333")

    # throughput interval (one block width) between consecutive event completions
    y_top = N_STAGES + 0.15
    ax.annotate("", xy=((N_EVENTS)*W, y_top), xytext=((N_EVENTS-1)*W, y_top),
                arrowprops=dict(arrowstyle="<->", color="#a31515", lw=1.8))
    ax.text((N_EVENTS-0.5)*W, y_top+0.15,
            "throughput interval ≈ 0.13 ms  →  7,549 ev/s (measured)",
            ha="center", va="bottom", fontsize=9.5, weight="bold", color="#a31515")

    # legend: events
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=EVENT_COLORS[e], label=f"event {e+1}") for e in range(N_EVENTS)],
              loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9, frameon=False,
              title="events in flight")
    ax.set_title("Pipelining schematic: 6 attention blocks overlap across events",
                 fontsize=13, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path+".png", bbox_inches="tight")
    fig.savefig(path+".pdf", bbox_inches="tight")
    print(f"wrote {path}.{{png,pdf}}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figs"); os.makedirs(out, exist_ok=True)
    fig(os.path.join(out, "pipeline_schematic"))
    print("Done.")
