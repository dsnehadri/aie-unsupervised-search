#!/usr/bin/env python
"""Quantization error of the AIE int16 attention vs the PyTorch float golden,
per event sample. Data: 20-event x86sim run (bit-exact model of the hardware
tiles) compared against phase3_export golden tensors -- the same comparison
check_attn_outputs.py scores, plotted per sample.

Per event: small dots = per-element |error| (unpadded rows); bold line =
per-event max. 1 LSB reference shows the single-rounding floor; everything
above it is accumulation through the block's gemms/softmax/LN chain."""
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

def block_errors(fname, gold_name, rows, scale, use_mask):
    per = rows * E_DIM
    raw = parse_plio(os.path.join(TB, "x86simulator_output/data", fname))
    gold = np.load(os.path.join(TV, gold_name))
    masks = np.load(os.path.join(TB, "data/padding_mask_event.npy")).astype(bool)
    elems, mx = [], []
    for ev in range(NEV):
        comp = raw[ev*per:(ev+1)*per].astype(np.float64).reshape(rows, E_DIM) / scale
        g = gold[ev].reshape(rows, E_DIM)
        err = np.abs(comp - g)
        if use_mask:
            err = err[~masks[ev]]
        elems.append(err.ravel())
        mx.append(err.max())
    return elems, np.array(mx)

BLOCKS = [  # name, output file, golden, rows, dequant scale, mask padded rows
    ("obj self-attn",   "obj_x_out_L0.txt",   "stage3_layer0_post_obj_selfattn.npy",  N_MAX, 2048.0, True,  "#2a78d6"),
    ("cand self-attn",  "cand_c_out_L0.txt",  "stage3_layer0_post_cand_selfattn.npy", T_DIM,  512.0, False, "#eb6834"),
    ("cross attn",      "cross_x_out_L0.txt", "stage3_layer0_post_cross_attn.npy",    N_MAX, 2048.0, True,  "#1baf7a"),
]

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(11.5, 5.2))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
rng = np.random.default_rng(0)

for name, f, g, rows, scale, um, col in BLOCKS:
    elems, mx = block_errors(f, g, rows, scale, um)
    for ev, e in enumerate(elems):  # per-element cloud, jittered
        ax.plot(ev + rng.uniform(-0.18, 0.18, e.size), e, ".", color=col,
                ms=2.5, alpha=0.28, markeredgewidth=0, zorder=2)
    ax.plot(range(NEV), mx, "-o", color=col, lw=2, ms=6.5, zorder=4, label=name)
    ax.annotate(name, xy=(NEV - 1, mx[-1]), xytext=(NEV - 0.6, mx[-1]),
                color=col, fontsize=10, va="center", weight="bold")

lsb = 1.0 / 2048
ax.axhline(lsb, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(0.1, lsb * 1.4, "1 LSB (obj/cross quantum, 1/2048)", fontsize=8.5, color=INK2)

ax.set_xlabel("event sample index", fontsize=11, color=INK)
ax.set_ylabel("|AIE int16  −  PyTorch float32|", fontsize=11, color=INK)
ax.set_title("Quantization error of the AIE attention blocks, per event\n"
             "dots = per-element error · line = per-event max · "
             "x86sim (bit-exact model of the hardware tiles) vs float golden, 20 events",
             fontsize=11, loc="left", color=INK)
ax.set_xticks(range(0, NEV, 2))
ax.set_xlim(-0.5, NEV + 1.6)
ax.set_ylim(0, 0.072)
ax.grid(alpha=0.15, axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK2)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
ax.text(0.995, 0.02, "pass tolerance 0.5 is ~7× above this axis · all 20 events pass",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/quantization_error.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURF)
print("saved", out)
for name, f, g, rows, scale, um, col in BLOCKS:
    elems, mx = block_errors(f, g, rows, scale, um)
    alle = np.concatenate(elems)
    print(f"{name:15s} per-elem median={np.median(alle):.5f}  p95={np.percentile(alle,95):.5f}  "
          f"per-event max: mean={mx.mean():.5f} worst={mx.max():.5f}")
