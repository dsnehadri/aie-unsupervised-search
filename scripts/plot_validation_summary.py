#!/usr/bin/env python
"""Validation summary: per-event max error vs PyTorch for the obj
self-attention block output, four builds of the SAME source -- PL and AIE,
each in float32 (unquantized) and int16 (deployed). One dot per event.

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

# ---- strip plot: 4 configurations, one dot per event ----------------------
groups = [
    ("PL\nfloat32",  pl_flt,  PL_C,  "none"),
    ("AIE\nfloat32", aie_flt, AIE_C, "none"),
    ("PL\nint16",    pl_fx,   PL_C,  "full"),
    ("AIE\nint16",   aie_fx,  AIE_C, "full"),
]

rng = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for gi, (label, vals, color, fill) in enumerate(groups):
    x = gi + rng.uniform(-0.14, 0.14, len(vals))
    ax.plot(x, vals, "o", ms=6, alpha=0.75, color=color,
            markerfacecolor=(color if fill == "full" else "white"),
            markeredgecolor=color, markeredgewidth=1.2, zorder=3)
    med = np.median(vals)
    ax.hlines(med, gi - 0.26, gi + 0.26, color="black", lw=1.6, zorder=4)
    ax.annotate(f"{med:.2g}%", (gi + 0.30, med), va="center",
                fontsize=10, weight="bold")

ax.set_yscale("log")
ax.set_ylim(1e-5, 50)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=11.5)
ax.set_xlim(-0.5, len(groups) - 0.25)
ax.set_ylabel("per-event max |error| vs PyTorch  [% of activation RMS]",
              fontsize=11)
ax.set_title("Validation vs PyTorch: same source, two arithmetics\n"
             "(obj self-attention block output; one dot per event; "
             "bar = median)", fontsize=12.5)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.grid(axis="x", visible=False)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/validation_summary.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
for label, vals, *_ in groups:
    print(f"{label.replace(chr(10),' '):14s} N={len(vals):3d}  "
          f"median {np.median(vals):.3g}%  max {vals.max():.3g}%")
