#!/usr/bin/env python
"""The unquantized-reference proof, per event: the SAME C++ obj-attention-block
source compiled two ways -- int16 fixed-point (default) and FLOAT_DATAPATH
(unquantized) -- each compared against PyTorch float32 on identical events.
Log y: the float build sits at float32 op-reordering noise (~1e-4 %), four
orders of magnitude below the fixed-point build's quantization floor,
demonstrating that the implementation's logic is exact and ALL fixed-point
deviation is quantization."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "/home/snehadri/aie_scratch_save_20260810"
N = 2000
gold = np.load(f"{SAVE}/m_obj2000.npy")               # PyTorch float32
mask = np.load(f"{SAVE}/m_mask2000.npy")
fx  = np.fromfile(f"{SAVE}/M_fx_s3.bin",  dtype=np.float32).reshape(N, 12, 16)
flt = np.fromfile(f"{SAVE}/M_flt_s3.bin", dtype=np.float32).reshape(N, 12, 16)
rms = np.sqrt(np.mean(gold**2))

def per_event(arr):
    d = np.abs(arr - gold)
    for i in range(N): d[i][mask[i]] = 0
    return 100.0 * d.reshape(N, -1).max(1) / rms      # % of activation RMS

e_fx, e_flt = per_event(fx), per_event(flt)

GROUPS = [("PL\nfloat32", e_flt, "#d62728", "none"),
          ("PL\nint16",   e_fx,  "#d62728", "full")]
YLABEL = "Per-event maximum error vs PyTorch\n[% of activation RMS]"
OUT = "/home/snehadri/repos/aie-unsupervised-search/figs/unquantized_proof.png"

# ---- error-bar plot in the validation-summary style ----------------------
fig, ax = plt.subplots(figsize=(5.2, 5.2))
for gi, (label, vals, color, fill) in enumerate(GROUPS):
    med = np.median(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    ax.errorbar(gi, med, yerr=[[med - lo], [hi - med]],
                fmt="o", ms=9, capsize=6, capthick=1.6, elinewidth=1.6,
                color=color, markerfacecolor=(color if fill == "full" else "white"),
                markeredgecolor=color, markeredgewidth=1.8, zorder=3)
ax.set_yscale("log")
ax.set_ylim(1e-5, 50)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g[0] for g in GROUPS], fontsize=12)
ax.set_xlim(-0.6, len(GROUPS) - 0.4)
ax.set_ylabel(YLABEL, fontsize=12)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.grid(axis="x", visible=False)
ax.tick_params(which="both", direction="in", right=True, top=True)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", OUT)
for label, vals, *_ in GROUPS:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{label.replace(chr(10),' '):16s} median {np.median(vals):.3g}%  "
          f"95% [{lo:.3g}, {hi:.3g}]  max {vals.max():.3g}%")
