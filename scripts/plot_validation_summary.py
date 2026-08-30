#!/usr/bin/env python
"""Validation summary: per-event max error vs PyTorch for the obj
self-attention block output, four builds of the SAME source -- PL and AIE,
each in float32 (unquantized) and int16 (deployed). Median with
16th-84th percentile error bars.

Data:
  PL  -- float_stage_check dumps (fx/flt_stage3.bin) vs torch_obj100.npy
  AIE -- x86sim all-blocks runs (float_aie_errors.json / int16_aie_errors.json)
         vs stage3_layer0_post_obj_selfattn goldens
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "/home/snehadri/aie_scratch_save_20260810"
TB = "/home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb"
TV = "/home/snehadri/repos/unsupervised-search/phase3_export_retrained/test_vectors"
PL_C, AIE_C = "#d62728", "#1f77b4"

# ---- PL: obj block stage dumps, 100 events --------------------------------
N_PL = 100
gold = np.load(f"{SAVE}/torch_obj100.npy")
mask = np.load(f"{SAVE}/mask100.npy")
rms_pl = np.sqrt(np.mean(gold ** 2))
def pl_pct(binfile):
    arr = np.fromfile(binfile, dtype=np.float32).reshape(N_PL, 12, 16)
    d = np.abs(arr - gold)
    for i in range(N_PL):
        d[i][mask[i]] = 0
    return 100.0 * d.reshape(N_PL, -1).max(1) / rms_pl
pl_fx, pl_flt = pl_pct(f"{SAVE}/fx_stage3.bin"), pl_pct(f"{SAVE}/flt_stage3.bin")

# ---- AIE: obj_L0 block from the x86sim all-blocks runs, 20 events ---------
with open(f"{TB}/float_aie_errors.json") as f:
    aie_flt_abs = np.array(json.load(f)["obj_L0"])
with open(f"{TB}/int16_aie_errors.json") as f:
    aie_fx_abs = np.array(json.load(f)["obj_L0"])
N_AIE = len(aie_flt_abs)
g = np.load(f"{TV}/stage3_layer0_post_obj_selfattn.npy")[:N_AIE]
rms_aie = np.sqrt(np.mean(g ** 2))
aie_flt = 100.0 * aie_flt_abs / rms_aie
aie_fx = 100.0 * aie_fx_abs / rms_aie

# ---- error-bar plot: 4 configurations ------------------------------------
groups = [
    ("PL\nfloat32",  pl_flt,  PL_C,  "none"),
    ("AIE\nfloat32", aie_flt, AIE_C, "none"),
    ("PL\nint16",    pl_fx,   PL_C,  "full"),
    ("AIE\nint16",   aie_fx,  AIE_C, "full"),
]

fig, ax = plt.subplots(figsize=(7.2, 5.2))
for gi, (label, vals, color, fill) in enumerate(groups):
    med = np.median(vals)
    lo, hi = np.percentile(vals, [16, 84])
    ax.errorbar(gi, med, yerr=[[med - lo], [hi - med]],
                fmt="o", ms=9, capsize=6, capthick=1.6, elinewidth=1.6,
                color=color, markerfacecolor=(color if fill == "full" else "white"),
                markeredgecolor=color, markeredgewidth=1.8, zorder=3)
    ax.annotate(f"{med:.2g}%", (gi + 0.13, med), va="center", ha="left",
                fontsize=10.5, weight="bold")

ax.set_yscale("log")
ax.set_ylim(1e-5, 50)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=12)
ax.set_xlim(-0.5, len(groups) - 0.35)
ax.set_ylabel("Per-event maximum error vs PyTorch\n[% of activation RMS]",
              fontsize=12)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.grid(axis="x", visible=False)
ax.tick_params(which="both", direction="in", right=True, top=True)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/validation_summary.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
for label, vals, *_ in groups:
    lo, hi = np.percentile(vals, [16, 84])
    print(f"{label.replace(chr(10),' '):14s} N={len(vals):3d}  "
          f"median {np.median(vals):.3g}%  16-84% [{lo:.3g}, {hi:.3g}]  "
          f"max {vals.max():.3g}%")
