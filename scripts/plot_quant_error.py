#!/usr/bin/env python
"""Quantization error of the AIE int16 attention vs the PyTorch float32 golden,
in the simple functional-verification style: signed difference per sample,
zero reference line, max-abs box. Samples = all attention output elements
(unpadded) of 20 events from the x86sim run (bit-exact model of the tiles),
concatenated obj | cand | cross. Difference is relative: % of the event's
golden activation RMS."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TB = "/home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb"
TV = "/home/snehadri/repos/unsupervised-search/phase3_export/test_vectors"
N_MAX, E_DIM, T_DIM = 12, 16, 3
NEV = 20

def parse_plio(path):
    vals = []
    for line in open(path):
        t = line.split()
        if not t or t[0] == "T":
            continue
        vals += [int(x) for x in t]
    return np.array(vals, dtype=np.int16)

def block_signed_rel(fname, gold_name, rows, scale, use_mask):
    per = rows * E_DIM
    raw = parse_plio(os.path.join(TB, "x86simulator_output/data", fname))
    gold = np.load(os.path.join(TV, gold_name))
    masks = np.load(os.path.join(TB, "data/padding_mask_event.npy")).astype(bool)
    out = []
    for ev in range(NEV):
        comp = raw[ev*per:(ev+1)*per].astype(np.float64).reshape(rows, E_DIM) / scale
        g = gold[ev].reshape(rows, E_DIM)
        if use_mask:
            comp, g = comp[~masks[ev]], g[~masks[ev]]
        rms = np.sqrt(np.mean(g**2))
        out.append(100.0 * (comp - g).ravel() / rms)
    return np.concatenate(out)

obj   = block_signed_rel("obj_x_out_L0.txt",   "stage3_layer0_post_obj_selfattn.npy",  N_MAX, 2048.0, True)
cand  = block_signed_rel("cand_c_out_L0.txt",  "stage3_layer0_post_cand_selfattn.npy", T_DIM,  512.0, False)
cross = block_signed_rel("cross_x_out_L0.txt", "stage3_layer0_post_cross_attn.npy",    N_MAX, 2048.0, True)
diff = np.concatenate([obj, cand, cross])
x = np.arange(diff.size)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, diff, "o", color="#3b3bcc", ms=2.5, alpha=0.6,
        markeredgecolor="k", markeredgewidth=0.15, label="Difference")
ax.axhline(0, color="red", lw=1, ls="--", label="Zero Diff")

# block region separators
b1, b2 = obj.size, obj.size + cand.size
for xb in (b1, b2):
    ax.axvline(xb, color="gray", lw=0.8, ls=":", alpha=0.7)
for xc, name in ((b1/2, "obj self-attn"), (b1 + cand.size/2, "cand"), (b2 + cross.size/2, "cross attn")):
    ax.text(xc, -7.2, name, ha="center", fontsize=10, color="dimgray")

ax.text(0.015, 0.965, f"Max Abs Diff: {np.abs(diff).max():.1f}%   (int16 quantization)",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray"))
ax.set_xlabel("Sample Index", fontsize=12)
ax.set_ylabel("Actual - Expected  [% of activation RMS]", fontsize=12)
ax.set_title("Quantization Error: AIE int16 Attention vs PyTorch float32 (x86sim, 20 events)", fontsize=14)
ax.set_ylim(-8, 8)
ax.grid(ls="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=11)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/quantization_error.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
print(f"samples={diff.size}  max abs = {np.abs(diff).max():.2f}%  "
      f"mean abs = {np.abs(diff).mean():.2f}%  bias = {diff.mean():+.3f}%")
