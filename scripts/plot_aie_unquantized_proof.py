#!/usr/bin/env python
"""AIE unquantized-reference proof, per event: the SAME AIE kernel/graph
source compiled two ways -- int16 (deployed, x86sim = bit-exact for the
integer kernels) and FLOAT_AIE (float32) -- each attention block fed exact
PyTorch inputs and compared against the PyTorch float32 output of that block.
Per event we plot the WORST block (max over the 6 attention blocks: obj/cand/
cross x L0/L1). Log y: the float build sits at float32 op-reordering noise,
orders of magnitude below the int16 build's quantization floor -> the AIE
kernel logic is exact and all int16 deviation is quantization."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TB = "/home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb"
TV = "/home/snehadri/repos/unsupervised-search/phase3_export_retrained/test_vectors"

with open(f"{TB}/float_aie_errors.json") as f:
    e_flt = json.load(f)
with open(f"{TB}/int16_aie_errors.json") as f:
    e_fx = json.load(f)

blocks = sorted(e_flt.keys())
N = len(e_flt[blocks[0]])

# normalize by the activation RMS of each block's golden output (same
# convention as figs/unquantized_proof.png)
rms = {}
gold_files = {
    "obj_L0":   "stage3_layer0_post_obj_selfattn.npy",
    "cand_L0":  "stage3_layer0_post_cand_selfattn.npy",
    "cross_L0": "stage3_layer0_post_cross_attn.npy",
    "obj_L1":   "stage3_layer1_post_obj_selfattn.npy",
    "cand_L1":  "stage3_layer1_post_cand_selfattn.npy",
    "cross_L1": "stage3_layer1_post_cross_attn.npy",
}
for b, fn in gold_files.items():
    g = np.load(f"{TV}/{fn}")[:N]
    rms[b] = float(np.sqrt(np.mean(g ** 2)))

def worst_block_pct(errs):
    # per event: max over blocks of (max abs err / that block's golden RMS)
    out = np.zeros(N)
    for b in blocks:
        e = 100.0 * np.array(errs[b][:N]) / rms[b]
        out = np.maximum(out, e)
    return out

p_fx, p_flt = worst_block_pct(e_fx), worst_block_pct(e_flt)
x = np.arange(N)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, p_fx, "o", color="#3b3bcc", ms=6, alpha=0.85,
        markeredgecolor="k", markeredgewidth=0.3, label="int16 fixed-point (deployed)")
ax.plot(x, p_flt, "o", color="#1baf7a", ms=6, alpha=0.85,
        markeredgecolor="k", markeredgewidth=0.3, label="float32 build (same source)")
ax.set_yscale("log")

ax.axhline(np.median(p_fx), color="#3b3bcc", lw=1, ls=":", alpha=0.7)
ax.axhline(np.median(p_flt), color="#1baf7a", lw=1, ls=":", alpha=0.7)
ax.text(0.015, 0.965,
        f"median per-event worst-block error:  int16 {np.median(p_fx):.2f}%   "
        f"float32 {np.median(p_flt):.1e}%   (~{np.median(p_fx)/np.median(p_flt):,.0f}× apart)\n"
        "float32 level = op-reordering noise vs PyTorch → AIE kernel logic exact; "
        "all int16 deviation is quantization",
        transform=ax.transAxes, va="top", fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.set_xlabel("Event Index", fontsize=12)
ax.set_ylabel("per-event max |error| vs PyTorch  [% of activation RMS]", fontsize=11)
ax.set_title("Unquantized Reference: same AIE kernels, two arithmetics, vs PyTorch\n"
             "(worst of the 6 attention blocks per event, x86sim, retrained weights, "
             f"{N} events)", fontsize=12.5)
ax.grid(ls="--", alpha=0.4)
ax.legend(loc="center right", fontsize=10)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/aie_unquantized_proof.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
print(f"int16:  median {np.median(p_fx):.3f}%  max {p_fx.max():.2f}%")
print(f"float:  median {np.median(p_flt):.2e}%  max {p_flt.max():.2e}%")
for b in blocks:
    print(f"  {b:<10s} int16 max_abs={max(e_fx[b]):.4f}  float max_abs={max(e_flt[b]):.2e}")
