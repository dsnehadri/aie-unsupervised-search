#!/usr/bin/env python3
"""Log host CPU temperatures and package power via lm-sensors.

Usage:
    sensor_log.py -o run.csv -i 1.0

Columns: iso_time, elapsed_s, pkg0_c, pkg1_c, core_max_c, power_w
"""
import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from datetime import datetime


PKG_RE = re.compile(r"^Package id (\d+):")
TEMP_RE = re.compile(r"^\s*temp\d+_input:\s*([\d.]+)")
POW_RE = re.compile(r"^\s*power\d+_average:\s*([\d.]+)")


def sample():
    """Return (pkg0_c, pkg1_c, core_max_c, power_w) from `sensors -u`."""
    out = subprocess.run(
        ["sensors", "-u"], capture_output=True, text=True, check=False
    ).stdout

    pkg = {0: None, 1: None}
    core_max = None
    power = None
    current_pkg = None
    in_pkg_section = False

    lines = out.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = PKG_RE.match(line)
        if m:
            current_pkg = int(m.group(1))
            in_pkg_section = True
            i += 1
            continue
        # power
        m = POW_RE.match(line)
        if m and power is None:
            power = float(m.group(1))
        # temp under a Package id section: first temp belongs to package
        if in_pkg_section:
            m = TEMP_RE.match(line)
            if m:
                pkg[current_pkg] = float(m.group(1))
                in_pkg_section = False
        else:
            # other temps inside coretemp-isa-* are per-core
            m = TEMP_RE.match(line)
            if m:
                t = float(m.group(1))
                if core_max is None or t > core_max:
                    core_max = t
        # section boundary on blank/new label group
        if line == "" or (line and not line.startswith(" ") and ":" not in line):
            current_pkg = None
            in_pkg_section = False
        i += 1
    return pkg[0], pkg[1], core_max, power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-i", "--interval", type=float, default=1.0)
    ap.add_argument("-d", "--duration", type=float, default=None,
                    help="if set, run for this many seconds and exit")
    args = ap.parse_args()

    stop = {"flag": False}
    def _stop(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso_time", "elapsed_s",
                    "pkg0_c", "pkg1_c", "core_max_c", "power_w"])
        next_t = t0
        while not stop["flag"]:
            now = time.time()
            iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
            elapsed = now - t0
            try:
                p0, p1, cmax, pw = sample()
            except Exception as e:
                print(f"sample error: {e}", file=sys.stderr)
                p0 = p1 = cmax = pw = None
            w.writerow([
                iso, f"{elapsed:.3f}",
                "" if p0 is None else f"{p0:.2f}",
                "" if p1 is None else f"{p1:.2f}",
                "" if cmax is None else f"{cmax:.2f}",
                "" if pw is None else f"{pw:.2f}",
            ])
            fh.flush()
            if args.duration is not None and elapsed >= args.duration:
                break
            next_t += args.interval
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
