#!/usr/bin/env python3
"""Fold ON/OFF modulation cycles: per-rail dP and per-sensor dT with std errors.

For each cycle: delta = mean(interior of ON) - mean(interiors of flanking OFFs).
Interior = skip SETTLE seconds after each transition (xclbin load + thermal lag).
"""
import csv, sys, math, statistics

SETTLE_ON = 35    # s: skip xclbin-load/graph-open transient at ON start
SETTLE_OFF = 20   # s: skip thermal decay at OFF start

def main(csv_path, phase_path):
    rows = list(csv.DictReader(open(csv_path)))
    ons = []          # (start, end) epochs
    for line in open(phase_path):
        p = line.split()
        if p[0] == "on_start":
            ons.append([int(p[1]), None])
        elif p[0] == "on_end":
            ons[-1][1] = int(p[1])
    ons = [(s, e) for s, e in ons if e is not None]
    keys = [k for k in rows[0] if k != "epoch"]
    epochs = [float(r["epoch"]) for r in rows]

    def win_mean(key, lo, hi):
        xs = [float(r[key]) for r, t in zip(rows, epochs)
              if lo <= t < hi and float(r[key]) == float(r[key])]
        return statistics.mean(xs) if xs else float("nan")

    deltas = {k: [] for k in keys}
    for i, (s, e) in enumerate(ons):
        off_hi = ons[i + 1][0] if i + 1 < len(ons) else max(epochs)
        for k in keys:
            on_m = win_mean(k, s + SETTLE_ON, e)
            off_m = win_mean(k, e + SETTLE_OFF, off_hi)
            if on_m == on_m and off_m == off_m:
                deltas[k].append(on_m - off_m)

    print(f"cycles used: {len(deltas[keys[0]])}")
    print(f"{'signal':16s} {'delta':>10s} {'SE':>8s}")
    for k in keys:
        d = deltas[k]
        if len(d) < 2:
            continue
        m = statistics.mean(d)
        se = statistics.stdev(d) / math.sqrt(len(d))
        unit = "C" if not k.endswith("_W") else "W"
        if abs(m) > 3 * se or k in ("versal", "aie", "total_W") or abs(m) > 0.01:
            print(f"{k:16s} {m:+10.4f} {se:8.4f}  {unit}{'  *' if abs(m) > 3*se else ''}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
