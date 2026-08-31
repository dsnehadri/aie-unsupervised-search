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

GROUPS = [("AIE\nfloat32", p_flt, "#1f77b4", "none"),
          ("AIE\nint16",   p_fx,  "#1f77b4", "full")]
YLABEL = ("Per-event maximum error vs PyTorch\n"
          "worst of six attention blocks  [% of activation RMS]")
OUT = "/home/snehadri/repos/aie-unsupervised-search/figs/aie_unquantized_proof.png"

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
