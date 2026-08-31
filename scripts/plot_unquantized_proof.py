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
x = np.arange(N)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, e_fx, "o", color="#3b3bcc", ms=5, alpha=0.8,
        markeredgecolor="k", markeredgewidth=0.3, label="int16 fixed-point (deployed)")
ax.plot(x, e_flt, "o", color="#1baf7a", ms=5, alpha=0.8,
        markeredgecolor="k", markeredgewidth=0.3, label="float32 build (same source)")
ax.set_yscale("log")
ax.set_ylim(1e-6, 50)

ax.axhline(np.median(e_fx), color="#3b3bcc", lw=1, ls=":", alpha=0.7)
ax.axhline(np.median(e_flt), color="#1baf7a", lw=1, ls=":", alpha=0.7)
ax.text(0.015, 0.965,
        f"median per-event max error:  int16 {np.median(e_fx):.2f}%   "
        f"float32 {np.median(e_flt):.1e}%   (~{np.median(e_fx)/np.median(e_flt):,.0f}× apart)\n"
        "float32 level = op-reordering noise vs PyTorch → logic exact; "
        "all int16 deviation is quantization",
        transform=ax.transAxes, va="top", fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.set_xlabel("Event Index", fontsize=12)
ax.set_ylabel("per-event max |error| vs PyTorch  [% of activation RMS]", fontsize=11)
ax.set_title("Unquantized Reference: same obj-attention source, two arithmetics, vs PyTorch\n"
             "(obj self-attention block output, retrained weights, 2000 events)", fontsize=12.5)
ax.grid(ls="--", alpha=0.4)
ax.legend(loc="center right", fontsize=10)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/unquantized_proof.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
print(f"int16:  median {np.median(e_fx):.3f}%  max {e_fx.max():.2f}%")
print(f"float:  median {np.median(e_flt):.2e}%  max {e_flt.max():.2e}%")
