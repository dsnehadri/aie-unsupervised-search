#!/usr/bin/env python3
"""Analyze VCK190 thermal/power run CSV + phase log; print summary, emit charts."""
import csv, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = os.path.dirname(os.path.abspath(__file__))

# palette (dataviz reference, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

def load(csv_path, phase_path):
    rows = list(csv.DictReader(open(csv_path)))
    phases = {}
    for line in open(phase_path):
        parts = line.split()
        phases[parts[0]] = int(parts[1])
    return rows, phases

def col(rows, name):
    return [float(r[name]) for r in rows]

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")

def main(csv_path, phase_path):
    rows, ph = load(csv_path, phase_path)
    t0 = float(rows[0]["epoch"])
    t = [float(r["epoch"]) - t0 for r in rows]
    load_s, load_e = ph["load_start"] - t0, ph["load_end"] - t0

    rails = [k[:-2] for k in rows[0] if k.endswith("_W") and k != "total_W"]
    idle_rows = [r for r in rows if float(r["epoch"]) - t0 < load_s - 5]
    # steady-state load: last two-thirds of the load window
    ss_lo = load_s + (load_e - load_s) / 3
    ss_rows = [r for r in rows if ss_lo < float(r["epoch"]) - t0 < load_e - 5]

    print(f"samples={len(rows)} idle_n={len(idle_rows)} load_ss_n={len(ss_rows)}")
    print(f"load window: {load_s:.0f}..{load_e:.0f} s ({load_e-load_s:.0f} s)")
    print()
    print(f"{'rail':14s} {'idle W':>9s} {'load W':>9s} {'delta W':>9s}")
    deltas = {}
    for rail in rails + ["total"]:
        iw = mean([float(r[rail + "_W"]) for r in idle_rows])
        lw = mean([float(r[rail + "_W"]) for r in ss_rows])
        deltas[rail] = (iw, lw, lw - iw)
        print(f"{rail:14s} {iw:9.3f} {lw:9.3f} {lw-iw:+9.3f}")
    print()
    for tn in ("versal", "aie", "sysmon_max"):
        ti = mean([float(r[tn]) for r in idle_rows])
        tl = mean([float(r[tn]) for r in ss_rows])
        tmax = max(float(r[tn]) for r in rows)
        print(f"temp {tn:11s} idle={ti:6.2f}C  load_ss={tl:6.2f}C  max={tmax:6.2f}C")

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)

    def shade(ax):
        ax.axvspan(load_s, load_e, color=GRID, alpha=0.35, zorder=0)

    # chart 1: total power
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax); shade(ax)
    tot = col(rows, "total_W")
    ax.plot(t, tot, color=BLUE, linewidth=0.8, alpha=0.25)
    k = 30
    roll = [sum(tot[max(0, i - k + 1):i + 1]) / len(tot[max(0, i - k + 1):i + 1])
            for i in range(len(tot))]
    ax.plot(t, roll, color=BLUE, linewidth=2, label="30 s rolling mean")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax.set_ylabel("board power (W)", color=INK2, fontsize=9)
    ax.set_title("VCK190 total rail power — idle / sustained PL matmul load / cooldown",
                 color=INK, fontsize=10, loc="left")
    ax.annotate("load", xy=((load_s + load_e) / 2, ax.get_ylim()[1]),
                ha="center", va="top", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRATCH, "thermal_total_power.png"))

    # chart 2: temperatures
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax); shade(ax)
    for name, colr, lbl in (("versal", BLUE, "Versal die"),
                            ("aie", ORANGE, "AIE array"),
                            ("sysmon_max", AQUA, "sysmon peak-hold (max since boot)")):
        ax.plot(t, col(rows, name), color=colr, linewidth=2, label=lbl)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax.set_ylabel("temperature (°C)", color=INK2, fontsize=9)
    ax.set_title("VCK190 die temperatures during load run", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRATCH, "thermal_temps.png"))

    # chart 3: per-rail idle vs load (top movers)
    movers = sorted((r for r in rails), key=lambda r: -abs(deltas[r][2]))[:8]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ys = range(len(movers))
    h = 0.38
    ax.barh([y + h / 2 + 0.02 for y in ys], [deltas[r][0] for r in movers],
            height=h, color=BLUE, label="idle")
    ax.barh([y - h / 2 - 0.02 for y in ys], [deltas[r][1] for r in movers],
            height=h, color=ORANGE, label="load (steady state)")
    ax.set_yticks(list(ys), movers)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_xlabel("rail power (W)", color=INK2, fontsize=9)
    ax.set_title("Per-rail power: idle vs sustained load (top movers)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRATCH, "thermal_rails.png"))
    print("\ncharts written to", SCRATCH)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
