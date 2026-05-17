#!/usr/bin/env python3
# Parse Vitis HLS csynth reports and emit a markdown table of per-kernel
# latency = cycles * clock period, plus II for throughput-equivalent time.
#
# Usage: scripts/bench/summarize.py [reports_dir]
# Default reports_dir: scripts/bench/reports

import os
import re
import sys
from pathlib import Path

REPORTS = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/bench/reports")

TIMING_RE = re.compile(
    r"\|ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.?]+)\s*ns\|\s*([\d.]+)\s*ns\|"
)

# top-level latency table row (after "+ Latency:" "* Summary:")
LATENCY_RE = re.compile(
    r"\|\s*(\d+|\?)\s*\|\s*(\d+|\?)\s*\|\s*([\d.]+\s*[num]?s|\?)\s*\|\s*([\d.]+\s*[num]?s|\?)\s*\|\s*(\d+|\?)\s*\|\s*(\d+|\?)\s*\|\s*\S+\s*\|"
)


def parse_report(path: Path):
    text = path.read_text()
    # timing
    target = est = uncert = None
    m = TIMING_RE.search(text)
    if m:
        target = float(m.group(1))
        try:
            est = float(m.group(2))
        except ValueError:
            est = None
        uncert = float(m.group(3))

    # find the first latency summary block; the values right under "Latency (cycles) | Latency (absolute) | Interval | Pipeline Type"
    lat_block = None
    idx = text.find("+ Latency:")
    if idx >= 0:
        # take the next ~30 lines
        chunk = text[idx:idx + 2000]
        for m2 in LATENCY_RE.finditer(chunk):
            lat_block = m2.groups()
            break

    lat_min = lat_max = int_min = int_max = None
    abs_min = abs_max = None
    if lat_block:
        lat_min = lat_block[0]
        lat_max = lat_block[1]
        abs_min = lat_block[2].replace(" ", "")
        abs_max = lat_block[3].replace(" ", "")
        int_min = lat_block[4]
        int_max = lat_block[5]

    return {
        "target_ns": target,
        "est_ns": est,
        "uncert_ns": uncert,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "abs_min": abs_min,
        "abs_max": abs_max,
        "int_min": int_min,
        "int_max": int_max,
    }


def fmax_mhz(period_ns):
    if period_ns is None or period_ns == 0:
        return None
    return 1000.0 / period_ns


def time_ns(cycles, period_ns):
    if cycles is None or cycles == "?" or period_ns is None:
        return None
    return int(cycles) * period_ns


def fmt_time_ns(ns):
    if ns is None:
        return "?"
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.3f} us"
    return f"{ns:.1f} ns"


def main():
    if not REPORTS.is_dir():
        print(f"No reports dir: {REPORTS}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for rpt in sorted(REPORTS.glob("*_csynth.rpt")):
        # name format: <kernel>_<topfn>_csynth.rpt
        stem = rpt.stem.replace("_csynth", "")
        info = parse_report(rpt)
        rows.append((stem, info))

    # Build markdown
    print("# PL kernel speed summary (HLS csynth estimates)")
    print()
    print("Target clock: 5 ns (200 MHz). Fmax_est = 1 / (estimated period).")
    print()
    print("| Kernel | Est. period (ns) | Fmax_est (MHz) | Latency cycles (min/max) | Latency (min/max) | II cycles (min/max) | Throughput-eq time |")
    print("|---|---:|---:|---|---|---|---|")
    for stem, info in rows:
        period = info["est_ns"]
        fmax = fmax_mhz(period)
        lat_min_t = time_ns(info["lat_min"], period)
        lat_max_t = time_ns(info["lat_max"], period)
        ii_min_t = time_ns(info["int_min"], period)
        period_s = "?" if period is None else f"{period:.3f}"
        fmax_s = "?" if fmax is None else f"{fmax:.1f}"
        cyc = f"{info['lat_min']} / {info['lat_max']}"
        lat = f"{fmt_time_ns(lat_min_t)} / {fmt_time_ns(lat_max_t)}"
        ii = f"{info['int_min']} / {info['int_max']}"
        thr = fmt_time_ns(ii_min_t)
        print(f"| {stem} | {period_s} | {fmax_s} | {cyc} | {lat} | {ii} | {thr} |")


if __name__ == "__main__":
    main()
