#!/usr/bin/env python3
"""Parse aiesimulator --profile output for the attn_test build.

Reads:
  src/aie_stream/tb/aiesimulator_output/profile_funct_<col>_<row>.txt
  src/aie_stream/tb/Work/reports/aie_attn_test_mapping_analysis_report.txt

Emits:
  scripts/aie_profile_per_tile.json   — per-tile cycles, busy%, top funcs
  scripts/aie_profile_per_tile.csv    — flat table
"""
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TB   = REPO / "src/aie_stream/tb"
PROF = TB / "aiesimulator_output"
MAP  = TB / "Work/reports/aie_attn_test_mapping_analysis_report.txt"


def load_tile_mapping():
    """tile_to_kernel[(col,row)] = kernel_name."""
    out = {}
    text = MAP.read_text()
    for m in re.finditer(r"^i\d+:(\S+)\s+CR\((\d+),(\d+)\)", text, re.M):
        kernel, col, row = m.group(1), int(m.group(2)), int(m.group(3))
        out[(col, row)] = kernel
    return out


HEADER_RE = re.compile(r"Total cycle count\s*:\s*(\d+)")
REPORT_RE = re.compile(r"Report cycle count\s*:\s*(\d+)")
INSTR_RE  = re.compile(r"Total instruction count\s*:\s*(\d+)")
PMSZ_RE   = re.compile(r"Total size in program memory\s*:\s*(\d+)")
# Function row: "  calls  cyc_func  pct_func  ... cyc_funcdesc pct_funcdesc ... Function"
# The columns are space-separated numbers. The function name is the LAST token group.
FUNC_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)%\s+\S+\s+\S+\s+\S+\s+"
    r"(\d+)\s+([\d.]+)%\s+\S+\s+\S+\s+\S+\s+\d+\s+\d+\s+(\S.*)$",
    re.MULTILINE,
)
# Categorize hot functions by what they really represent.
# Order matters — first match wins.
def categorize(fn):
    f = fn.lower()
    if f.startswith("_main_init") or f == "main":
        return "main / init"
    if f.startswith("_fini") or "__cxa_finalize" in f:
        return "main / init"
    if "expf" in f:
        return "softmax (expf)"
    if "softmax" in f:
        return "softmax (expf)"
    if "layernorm" in f or "float_sqrtf" in f or "f32_sqrt" in f or "softfloat_approxrecip" in f:
        return "layernorm + sqrt"
    if f.startswith("f32_") or f.startswith("softfloat"):
        return "scalar-float helpers"
    if "_to_i32" in f or "_to_f32" in f:
        return "fixed↔float convert"
    if "gemm" in f or "mmul" in f or "_mac" in f:
        return "vector mmul"
    if "memset" in f or "memcpy" in f:
        return "memcpy / memset"
    # the kernel body itself (e.g. obj_attn_head_post_h0_L0)
    return "kernel body / glue"


def parse_profile(path):
    text = path.read_text()
    total  = int(HEADER_RE.search(text).group(1)) if HEADER_RE.search(text) else None
    report = int(REPORT_RE.search(text).group(1)) if REPORT_RE.search(text) else None
    instr  = int(INSTR_RE.search(text).group(1))  if INSTR_RE.search(text)  else None
    pmsz   = int(PMSZ_RE.search(text).group(1))   if PMSZ_RE.search(text)   else None

    # Each function appears twice in the report (once sorted by name, once by
    # cycles desc). Dedup by name, keeping the first occurrence.
    seen = set()
    funcs = []
    for m in FUNC_RE.finditer(text):
        calls       = int(m.group(1))
        cyc_func    = int(m.group(2))
        pct_func    = float(m.group(3))
        cyc_fd      = int(m.group(4))
        pct_fd      = float(m.group(5))
        name_blob   = m.group(6).strip()
        name = name_blob.split()[0]
        if name in seen:
            continue
        seen.add(name)
        funcs.append({
            "name": name,
            "calls": calls,
            "cyc_func": cyc_func,
            "pct_func": pct_func,
            "cyc_func_desc": cyc_fd,
            "pct_func_desc": pct_fd,
            "category": categorize(name),
        })

    # Sort by cycles spent IN the function (not including callees) descending
    funcs.sort(key=lambda x: -x["cyc_func"])

    return {
        "total_cycles":  total,
        "report_cycles": report,
        "busy_pct":      (100.0 * report / total) if (total and report) else None,
        "instructions":  instr,
        "pm_size":       pmsz,
        "functions":     funcs,
    }


def main():
    tile_to_kernel = load_tile_mapping()

    rows = []
    for path in sorted(PROF.glob("profile_funct_*.txt")):
        m = re.match(r"profile_funct_(\d+)_(\d+)\.txt", path.name)
        if not m:
            continue
        col, row = int(m.group(1)), int(m.group(2))
        kernel = tile_to_kernel.get((col, row), "(unmapped)")
        info = parse_profile(path)
        info["col"], info["row"], info["kernel"] = col, row, kernel
        rows.append(info)

    # Sort by row then col for layout
    rows.sort(key=lambda r: (r["col"], r["row"]))

    out_json = REPO / "scripts/aie_profile_per_tile.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_json}  ({len(rows)} tiles)")

    out_csv = REPO / "scripts/aie_profile_per_tile.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["col", "row", "kernel", "total_cyc", "report_cyc",
                    "busy_pct", "pm_size", "top_funcs"])
        for r in rows:
            top3 = "  •  ".join(
                f"{fn['name']}={fn['cyc_func']}({fn['pct_func']:.1f}%)"
                for fn in r["functions"][:3]
            )
            w.writerow([r["col"], r["row"], r["kernel"], r["total_cycles"],
                        r["report_cycles"],
                        f"{r['busy_pct']:.1f}" if r["busy_pct"] else "",
                        r["pm_size"], top3])
    print(f"wrote {out_csv}")

    # Quick text summary on stdout
    print("\nTop-5 busiest tiles:")
    for r in sorted(rows, key=lambda r: -(r["busy_pct"] or 0))[:5]:
        print(f"  {r['col']:2d}_{r['row']:1d}  {r['kernel']:35s}  "
              f"busy {r['busy_pct']:5.1f}%  PM {r['pm_size']:5d} B")
    print("\nBottom-5 quietest tiles:")
    for r in sorted(rows, key=lambda r: r["busy_pct"] or 100)[:5]:
        print(f"  {r['col']:2d}_{r['row']:1d}  {r['kernel']:35s}  "
              f"busy {r['busy_pct']:5.1f}%  PM {r['pm_size']:5d} B")


if __name__ == "__main__":
    main()
