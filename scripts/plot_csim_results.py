#!/usr/bin/env python3
"""parse vitis_hls csim logs and emit a two-panel plot of per-test max_err and rmse
against the pytorch reference, with each test's tolerance marked."""

import re
import os
import csv
from pathlib import Path
import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


def darken(color, amount=0.25):
    """return a slightly darker version of the given matplotlib color."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, l * (1.0 - amount))
    return colorsys.hls_to_rgb(h, l, s)

sns.set_theme(context="paper", style="whitegrid", palette="bright",
               font="Montserrat", font_scale=1.15)

# group components into a few categories for a less-noisy palette
GROUP_OF = {
    "attn_block":   "Attention",
    "autoencoder":  "Autoencoder",
    "cand_build":   "Candidate ops",
    "cand_lorentz": "Candidate ops",
    "embed_ffn":    "Per-token MLPs",
    "pairwise_mlp": "Per-token MLPs",
    "passwd_top":   "Full pipeline",
    "pl_stream":    "Full pipeline",
}
GROUP_ORDER = ["Attention", "Autoencoder", "Candidate ops",
               "Per-token MLPs", "Full pipeline"]
GROUP_COLORS = {
    "Attention":     "#A6C8FF",  # pastel blue
    "Autoencoder":   "#FFD3A5",  # pastel orange
    "Candidate ops": "#A8E6C3",  # pastel green
    "Per-token MLPs":"#D5C2F5",  # pastel purple
    "Full pipeline": "#F8B7CD",  # pastel pink
}
plt.rcParams.update({
    "font.family": "Montserrat",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Montserrat",
    "mathtext.it": "Montserrat:italic",
    "mathtext.bf": "Montserrat:bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 160,
})

# tests we explicitly skip in the plot (redundant or off-topic)
DROP_TESTS = {"mse_xloss", "latent_dist", "mse_crossed"}

# publication-quality display labels per component (one bar per src/ piece)
COMPONENT_LABEL = {
    "attn_block":   "Attention block (obj / cand / cross)",
    "autoencoder":  "Dual autoencoder",
    "cand_build":   "Candidate builder",
    "cand_lorentz": "Lorentz feature transform",
    "embed_ffn":    "Input embedding FFN",
    "pairwise_mlp": "Pairwise interaction MLP",
    "passwd_top":   "Full pipeline (MM-AXI)",
    "pl_stream":    "Full pipeline (PL stream)",
}

REPO = Path(__file__).resolve().parents[1]

# (component label, log path) -- one preferred log per project
LOGS = [
    ("attn_block",   REPO / "attn_block_proj/solution1/csim/report/attn_block_obj_csim.log"),
    ("autoencoder",  REPO / "autoencoder_proj/solution1/csim/report/dual_autoencoder_top_csim.log"),
    ("cand_build",   REPO / "cand_build_proj/solution1/csim/report/candidate_build_top_csim.log"),
    ("cand_lorentz", REPO / "cand_lorentz_proj/solution1/csim/report/cand_lorentz_top_csim.log"),
    ("embed_ffn",    REPO / "embed_ffn_proj/solution1/csim/report/embed_ffn_top_csim.log"),
    ("pairwise_mlp", REPO / "pairwise_mlp_proj/solution1/csim/report/pairwise_mlp_top_csim.log"),
    ("passwd_top",   REPO / "passwd_top_proj/solution1/csim/report/passwd_top_csim.log"),
    ("pl_stream",    REPO / "pl_stream/solution1/csim/report/pl_stream_top_csim.log"),
]


def parse_vector_log(text, fallback_name):
    """yield (test_name, max_err, rmse, tolerance) from `compare()`-style output.
       handles both forms: with a `testing <name>` header and without
       (some TBs call compare() with no name)."""
    # split into per-comparison blocks; each comparison contains a
    # "max absolute err:" line. walk forward to find rmse and tolerance.
    err_pat = re.compile(r"max absolute err:\s*([0-9.eE+-]+)")
    rmse_pat = re.compile(r"rmse:\s*([0-9.eE+-]+)")
    tol_pat = re.compile(r"(?:pass|fail):\s*max error\s*[0-9.eE+-]+\s*[<>]=?\s*tolerance\s*([0-9.eE+-]+)")
    name_pat = re.compile(r"testing\s+(\S+)")

    positions = [m.start() for m in err_pat.finditer(text)]
    for i, pos in enumerate(positions):
        # name: nearest "testing <name>" before this position, after the previous comparison
        prev = positions[i-1] if i > 0 else 0
        window = text[prev:pos]
        n = name_pat.findall(window)
        name = n[-1] if n else fallback_name

        # err/rmse/tol live in the block starting at pos
        end = positions[i+1] if i+1 < len(positions) else len(text)
        block = text[pos:end]
        em = err_pat.search(block)
        rm = rmse_pat.search(block)
        tm = tol_pat.search(block)
        if not (em and rm and tm):
            continue
        yield name, float(em.group(1)), float(rm.group(1)), float(tm.group(1))


def parse_scalar_log(text):
    """yield (test_name, err, tolerance_or_None) from `compare_scalar()`-style output:
       `<name> computed = <c> golden = <g> err <e> PASS`."""
    pat = re.compile(
        r"^\s*(\S+)\s+computed\s*=\s*([0-9.eE+-]+)\s+golden\s*=\s*([0-9.eE+-]+)\s+err\s+([0-9.eE+-]+)\s+(PASS|FAIL)",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        yield m.group(1), float(m.group(4)), None


def parse_autoencoder_array_log(text):
    """yield (name, max_err, None, None) for lines like
       ` c0_latent  max_abs_err = 0.000004 @ [1] PASS`."""
    pat = re.compile(
        r"^\s*(\S+)\s+max_abs_err\s*=\s*([0-9.eE+-]+)\s+@\s*\[\d+\]\s+(PASS|FAIL)",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        yield m.group(1), float(m.group(2)), None


def parse_stream_metric_log(text):
    """yield (name, abs_err, None) for the streaming table:
       `mse_loss  0.867435  0.525649  0.341786 PASS`."""
    pat = re.compile(
        r"^\s*(mse_loss|mse_crossed|latent_dist)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(PASS|FAIL)",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        yield m.group(1), float(m.group(4)), None


def collect():
    """returns list of dicts {component, test, max_err, rmse, tol, kind}."""
    rows = []
    for comp, path in LOGS:
        if not path.exists():
            print(f"warning: missing log {path}")
            continue
        text = path.read_text(errors="ignore")

        # try vector-compare format first
        seen = set()
        for name, mx, rm, tol in parse_vector_log(text, fallback_name=comp):
            # for multi-event logs (e.g. cand_lorentz with 3 events), keep worst case
            key = (comp, name)
            existing = next((r for r in rows if (r["component"], r["test"]) == key), None)
            if existing is None:
                rows.append({"component": comp, "test": name, "max_err": mx,
                             "rmse": rm, "tol": tol, "kind": "vector"})
            else:
                existing["max_err"] = max(existing["max_err"], mx)
                existing["rmse"] = max(existing["rmse"], rm)
            seen.add(name)

        # autoencoder array-style lines
        for name, err, _ in parse_autoencoder_array_log(text):
            if name in seen:
                continue
            rows.append({"component": comp, "test": name, "max_err": err,
                         "rmse": None, "tol": None, "kind": "array"})
            seen.add(name)

        # scalar-compare lines (passwd top, autoencoder losses)
        for name, err, _ in parse_scalar_log(text):
            if name in seen:
                continue
            rows.append({"component": comp, "test": name, "max_err": err,
                         "rmse": None, "tol": 0.01, "kind": "scalar"})
            seen.add(name)

        # streamed-pipeline metric table (passwd_stream / pl_stream).
        # TB uses TOL = 0.5 (relaxed because output is ap_fixed<16,5>); see
        # test_benches/pl_stream_tb.cpp:90
        for name, err, _ in parse_stream_metric_log(text):
            if name in seen:
                continue
            rows.append({"component": comp, "test": name, "max_err": err,
                         "rmse": None, "tol": 0.5, "kind": "stream"})
            seen.add(name)

    return rows


# small floor so log-scale bars are visible when err == 0
FLOOR = 1e-7


def plot(rows, out_png):
    # collapse to one bar per component: worst-case max_err across its tests
    comp_order = [c for c, _ in LOGS]
    rows = [r for r in rows if r["test"] not in DROP_TESTS]

    per_comp = {}
    for r in rows:
        c = r["component"]
        if c not in per_comp or r["max_err"] > per_comp[c]:
            per_comp[c] = r["max_err"]

    comps = [c for c in comp_order if c in per_comp]
    labels = [COMPONENT_LABEL.get(c, c) for c in comps]
    max_errs = [max(per_comp[c], FLOOR) for c in comps]
    face = "#A6C8FF"            # pastel blue for every bar
    edge = darken(face, 0.32)

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.5 * len(labels))))

    y = np.arange(len(labels))
    bars = ax.barh(y, max_errs, color=face, edgecolor=edge,
                   linewidth=1.4, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xscale("log")
    ax.set_xlabel(r"max $\,|\,$HLS $-$ PyTorch$\,|\,$  (log scale)")
    ax.set_ylabel("")

    sns.despine(ax=ax, left=True)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_png.with_suffix('.pdf')}")


def write_csv(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "test", "max_err", "rmse", "tolerance", "kind", "pass"])
        for r in rows:
            passed = (r["tol"] is None) or (r["max_err"] < r["tol"])
            w.writerow([r["component"], r["test"], r["max_err"],
                        r["rmse"] if r["rmse"] is not None else "",
                        r["tol"] if r["tol"] is not None else "",
                        r["kind"], "PASS" if passed else "FAIL"])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    rows = collect()
    if not rows:
        raise SystemExit("no rows parsed; check log paths")
    out_dir = REPO / "figs"
    out_dir.mkdir(exist_ok=True)
    plot(rows, out_dir / "csim_results.png")
    write_csv(rows, out_dir / "csim_results.csv")
