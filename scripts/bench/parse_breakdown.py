#!/usr/bin/env python3
# Walks each kernel's bench project, extracts the sub-instance latency breakdown
# from the "main" sub-function csynth report (not the _top wrapper), and writes
# a CSV plus a JSON suitable for plotting.

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# kernel -> (project dir name, main sub-function report stem, top-function report stem)
KERNELS = [
    ("embed_ffn",      "bench_embed_ffn_proj",      "embed_ffn",         "embed_ffn_top"),
    ("pairwise_mlp",   "bench_pairwise_mlp_proj",   "pairwise_mlp",      "pairwise_mlp_top"),
    ("attn_obj",       "bench_attn_obj_proj",       "attn_block_obj",    "attn_block_obj_top"),
    ("attn_cand",      "bench_attn_cand_proj",      "attn_block_cand",   "attn_block_cand_top"),
    ("attn_cross",     "bench_attn_cross_proj",     "attn_block_cross",  "attn_block_cross_top"),
    ("cand_build",     "bench_cand_build_proj",     "candidate_build_top","candidate_build_top"),
    ("cand_lorentz",   "bench_cand_lorentz_proj",   "cand_lorentz_top",  "cand_lorentz_top"),
    ("autoencoder",    "bench_autoencoder_proj",    "dual_autoencoder",  "dual_autoencoder_top"),
]

ROW_RE = re.compile(
    r"^\s*\|\s*([A-Za-z_][\w]*)\s*\|\s*([A-Za-z_][\w]*)\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|"
)
# we'll just parse using a simpler split since cell text varies

PIPE_SPLIT = re.compile(r"\s*\|\s*")
NUM_RE = re.compile(r"^\d+$")

TIMING_RE = re.compile(
    r"\|ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.?]+)\s*ns\|\s*([\d.]+)\s*ns\|"
)


def parse_period_ns(text):
    m = TIMING_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(2))
    except ValueError:
        return float(m.group(1))


def parse_top_latency(text):
    """Return (lat_min, lat_max) from the top * Summary: block."""
    idx = text.find("+ Latency:")
    if idx < 0:
        return None, None
    chunk = text[idx:idx + 1500]
    # find the first all-number row in the summary
    for line in chunk.splitlines():
        parts = [p.strip() for p in PIPE_SPLIT.split(line) if p.strip()]
        if len(parts) >= 6 and all(NUM_RE.match(p) for p in parts[:2]):
            return int(parts[0]), int(parts[1])
    return None, None


def parse_instances(text):
    """Return list of {name, module, min, max} from the * Instance: table inside + Latency: + Detail:."""
    idx = text.find("* Instance:")
    if idx < 0:
        return []
    # take from there up to next "* Loop:" or "==="
    end = text.find("* Loop:", idx)
    if end < 0:
        end = text.find("====", idx)
    chunk = text[idx:end if end > 0 else idx + 6000]

    rows = []
    for line in chunk.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 8:
            continue
        # Skip header rows
        if parts[0].lower().startswith("instance") or "latency" in parts[0].lower():
            continue
        # Need numeric min/max in cols 2,3
        if not (NUM_RE.match(parts[2]) and NUM_RE.match(parts[3])):
            continue
        rows.append({
            "instance": parts[0],
            "module": parts[1],
            "cycles_min": int(parts[2]),
            "cycles_max": int(parts[3]),
        })
    return rows


def parse_loops(text):
    """Return list of {name, lat_min, lat_max, trip_count, iter_latency}."""
    idx = text.find("* Loop:")
    if idx < 0:
        return []
    end = text.find("====", idx)
    chunk = text[idx:end if end > 0 else idx + 4000]
    rows = []
    for line in chunk.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 7:
            continue
        name = parts[0].lstrip("-+ ").strip()
        if not name or "Loop Name" in parts[0]:
            continue
        # numeric latencies
        if not (NUM_RE.match(parts[1]) and NUM_RE.match(parts[2])):
            continue
        # trip count is at col 6 (after iteration latency, ach II, target II)
        trip = parts[6] if NUM_RE.match(parts[6]) else None
        rows.append({
            "name": name,
            "lat_min": int(parts[1]),
            "lat_max": int(parts[2]),
            "iter_latency": parts[3],
            "trip_count": int(trip) if trip else None,
        })
    return rows


def main():
    out = []
    for kernel, projdir, sub_stem, top_stem in KERNELS:
        sub_rpt = REPO / projdir / "solution1" / "syn" / "report" / f"{sub_stem}_csynth.rpt"
        top_rpt = REPO / projdir / "solution1" / "syn" / "report" / f"{top_stem}_csynth.rpt"
        if not sub_rpt.exists() or not top_rpt.exists():
            print(f"missing report for {kernel}: {sub_rpt}", file=sys.stderr)
            continue

        top_text = top_rpt.read_text()
        sub_text = sub_rpt.read_text()
        period = parse_period_ns(top_text)
        top_min, top_max = parse_top_latency(top_text)
        instances = parse_instances(sub_text)
        loops = parse_loops(sub_text)

        out.append({
            "kernel": kernel,
            "period_ns": period,
            "top_cycles_min": top_min,
            "top_cycles_max": top_max,
            "instances": instances,
            "loops": loops,
        })

    # write JSON
    out_dir = REPO / "scripts" / "bench"
    (out_dir / "breakdown.json").write_text(json.dumps(out, indent=2))

    # write CSV: one row per (kernel, instance), plus total row
    with (out_dir / "breakdown.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "module", "cycles_min", "cycles_max", "us_min", "us_max", "is_total"])
        for k in out:
            p = k["period_ns"] or 5.0
            for inst in k["instances"]:
                w.writerow([
                    k["kernel"], inst["module"], inst["cycles_min"], inst["cycles_max"],
                    f"{inst['cycles_min']*p/1000.0:.3f}", f"{inst['cycles_max']*p/1000.0:.3f}", 0,
                ])
            w.writerow([
                k["kernel"], "TOTAL", k["top_cycles_min"], k["top_cycles_max"],
                f"{(k['top_cycles_min'] or 0)*p/1000.0:.3f}",
                f"{(k['top_cycles_max'] or 0)*p/1000.0:.3f}", 1,
            ])

    print(f"wrote {out_dir/'breakdown.json'} and {out_dir/'breakdown.csv'}")


if __name__ == "__main__":
    main()
