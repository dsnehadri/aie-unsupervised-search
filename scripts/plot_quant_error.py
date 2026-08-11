#!/usr/bin/env python
"""Quantization error of the AIE int16 attention vs the PyTorch float golden,
per event sample, as RELATIVE error: |int16 - float| as a percentage of the
event's golden activation RMS for that block. Data: 20-event x86sim run
(bit-exact model of the hardware tiles) vs phase3_export golden tensors.

Per block panel: shaded band = p5..p95 of per-element relative error,
solid line = median, marker line = per-event max."""
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

def block_rel_errors(fname, gold_name, rows, scale, use_mask):
    per = rows * E_DIM
    raw = parse_plio(os.path.join(TB, "x86simulator_output/data", fname))
    gold = np.load(os.path.join(TV, gold_name))
    masks = np.load(os.path.join(TB, "data/padding_mask_event.npy")).astype(bool)
    p5, p50, p95, mx = [], [], [], []
    for ev in range(NEV):
        comp = raw[ev*per:(ev+1)*per].astype(np.float64).reshape(rows, E_DIM) / scale
        g = gold[ev].reshape(rows, E_DIM)
        if use_mask:
            comp, g = comp[~masks[ev]], g[~masks[ev]]
        rms = np.sqrt(np.mean(g**2))              # event's activation scale
        rel = 100.0 * np.abs(comp - g).ravel() / rms
        p5.append(np.percentile(rel, 5)); p50.append(np.percentile(rel, 50))
        p95.append(np.percentile(rel, 95)); mx.append(rel.max())
    return map(np.array, (p5, p50, p95, mx))

BLOCKS = [
    ("obj self-attention",  "obj_x_out_L0.txt",   "stage3_layer0_post_obj_selfattn.npy",  N_MAX, 2048.0, True,  "#2a78d6"),
    ("cand self-attention", "cand_c_out_L0.txt",  "stage3_layer0_post_cand_selfattn.npy", T_DIM,  512.0, False, "#eb6834"),
    ("cross attention",     "cross_x_out_L0.txt", "stage3_layer0_post_cross_attn.npy",    N_MAX, 2048.0, True,  "#1baf7a"),
]

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True, sharey=True)
fig.patch.set_facecolor(SURF)
x = np.arange(NEV)

for ax, (name, f, g, rows, scale, um, col) in zip(axes, BLOCKS):
    ax.set_facecolor(SURF)
    p5, p50, p95, mx = block_rel_errors(f, g, rows, scale, um)
    ax.fill_between(x, p5, p95, color=col, alpha=0.18, lw=0, label="p5–p95 of elements")
    ax.plot(x, p50, "-", color=col, lw=2, label="median element")
    ax.plot(x, mx, "-o", color=col, lw=1.2, ms=5, alpha=0.75, label="worst element")
    ax.text(0.005, 0.86, name, transform=ax.transAxes, color=col,
            fontsize=11, weight="bold")
    # stats note: bottom-right in the cross panel (its band+worst line fill the top)
    ty = 0.06 if name.startswith("cross") else 0.86
    ax.text(0.995, ty, f"median ≈ {np.mean(p50):.1f}%   worst ≈ {mx.max():.0f}%",
            transform=ax.transAxes, color=INK2, fontsize=9, ha="right")
    ax.grid(alpha=0.15, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)

axes[0].set_title("Quantization error of the AIE attention blocks, per event\n"
                  "error as % of the event's golden activation RMS · "
                  "x86sim (bit-exact model of the tiles) vs PyTorch float32, 20 events",
                  fontsize=11, loc="left", color=INK)
axes[0].legend(frameon=False, fontsize=9, loc="upper right", ncol=3,
               bbox_to_anchor=(1.0, 1.02))
axes[-1].set_xlabel("event sample index", fontsize=11, color=INK)
axes[-1].set_xticks(range(0, NEV, 2))
axes[1].set_ylabel("relative error  [% of activation RMS]", fontsize=11, color=INK)

fig.tight_layout(h_pad=0.6)
out = "/home/snehadri/repos/aie-unsupervised-search/figs/quantization_error.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURF)
print("saved", out)
for name, f, g, rows, scale, um, col in BLOCKS:
    p5, p50, p95, mx = block_rel_errors(f, g, rows, scale, um)
    print(f"{name:20s} median={np.mean(p50):.2f}%  p95={np.mean(p95):.2f}%  worst={mx.max():.1f}%")
