#!/usr/bin/env python3
"""
gen_attn_inputs.py
------------------
Pre-sim helper for the AIE attention testbench.

Loads phase-3 .npy test vectors for one event, quantizes float -> int16 with
the Q4.11 scaling that matches data_t = ap_fixed<16,5> on the PL side, and
writes PLIO text files that the AIE simulator will consume.

PLIO text format (matches plio_64_bits with int16 data):
    Each line contains 4 int16 values in hex (4 chars each), space-separated.
    Each line is therefore one 64-bit PLIO transfer.
    Negative values are written as two's complement (e.g. -1 -> ffff).

Files written into ./data/:
    obj0:    obj_x_in_L0.txt, obj_wij_h0_L0.txt..h3_L0.txt
    cand0:   cand_c_in_L0.txt
    cross0:  cross_x_in_L0.txt, cross_c_in_L0.txt

Test vectors expected in <phase3>/test_vectors/:
    stage0_padding_mask.npy
    stage1_post_embedding.npy
    stage2_wij_post_mlp.npy
    stage3_layer0_candidates_embedded.npy
    stage3_layer0_post_obj_selfattn.npy
    stage3_layer0_post_cand_selfattn.npy

Mirrors the PL testbench's load_2d / load_padding_mask / wij replication logic.
"""

import argparse
import os
import sys
import numpy as np


# ----------------------------------------------------------------------------
# Architectural constants (must match attn_aie_types.h)
# ----------------------------------------------------------------------------
N_MAX = 12
E_DIM = 16
N_HEADS = 4
N_KV = 13          # N_MAX + 1 (bias_kv slot)
T_DIM = 3

# data_t = ap_fixed<16,5>  -> 11 fractional bits  -> scale = 2048
DATA_FRAC_BITS = 11
DATA_SCALE = 1 << DATA_FRAC_BITS  # 2048

# PLIO transfer width / element width
INT16S_PER_TRANSFER = 4   # plio_64_bits / 16 = 4


# ----------------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------------
def to_int16(x_float):
    """Float -> int16 with Q4.11 scaling, saturating."""
    q = np.round(np.asarray(x_float, dtype=np.float64) * DATA_SCALE)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.int16)


def write_plio_text(path, int16_arr):
    """Write a flat int16 array as a PLIO text file (4 signed-decimal int16s per line).

    aiesimulator's default PLIO parser reads tokens as signed decimal integers;
    hex tokens with letters (e.g. 'fcbd') fail to parse and silently starve the
    input stream, which deadlocks the graph. Always emit decimal.
    """
    flat = int16_arr.reshape(-1).astype(np.int16)

    # Pad to a multiple of INT16S_PER_TRANSFER so every line is a full transfer.
    pad = (-len(flat)) % INT16S_PER_TRANSFER
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int16)])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(0, len(flat), INT16S_PER_TRANSFER):
            row = flat[i:i + INT16S_PER_TRANSFER]
            f.write(" ".join(str(int(v)) for v in row) + "\n")
    print(f"  wrote {path}  ({len(flat)} int16s, {len(flat)//INT16S_PER_TRANSFER} transfers)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3", required=True,
                    help="Phase-3 export directory "
                         "(e.g. /home/snehadri/repos/unsupervised-search/phase3_export)")
    ap.add_argument("--event", type=int, default=0,
                    help="Event index to test (default: 0)")
    ap.add_argument("--data-dir", default="./data",
                    help="Where to write PLIO text files (default: ./data)")
    args = ap.parse_args()

    tv = os.path.join(args.phase3, "test_vectors")
    if not os.path.isdir(tv):
        sys.exit(f"test_vectors directory not found: {tv}")

    ev = args.event
    print(f"generating PLIO inputs for event {ev}")
    print(f"  test_vectors: {tv}")
    print(f"  output dir:   {args.data_dir}")

    def load_event(name, expected_shape):
        path = os.path.join(tv, name)
        arr = np.load(path)
        if arr.shape[0] <= ev:
            sys.exit(f"{name}: only {arr.shape[0]} events, requested ev={ev}")
        out = arr[ev]
        if out.shape != expected_shape:
            sys.exit(f"{name}: expected shape {expected_shape}, got {out.shape}")
        return out

    # padding mask for this event (True = padded jet, ignore)
    mask_arr = np.load(os.path.join(tv, "stage0_padding_mask.npy"))
    pad_mask = mask_arr[ev].astype(bool)
    if pad_mask.shape != (N_MAX,):
        sys.exit(f"padding_mask: expected shape ({N_MAX},), got {pad_mask.shape}")

    # ---- obj0 inputs --------------------------------------------------------
    print("\n[obj0]")
    x_obj = load_event("stage1_post_embedding.npy", (N_MAX, E_DIM))
    write_plio_text(os.path.join(args.data_dir, "obj_x_in_L0.txt"), to_int16(x_obj))

    # wij from pairwise MLP: shape (N_MAX, N_MAX); pad to (N_MAX, N_KV) with
    # zero in the bias_kv slot (col N_MAX). Same data feeds all 4 heads.
    #
    # IMPORTANT: PL passes padding_mask separately to attn_block_obj, but the
    # AIE kernel only consumes wij — so the deployed pipeline pre-bakes the
    # padding mask into wij upstream (large negative at padded key positions,
    # so softmax -> 0). We replicate that here to test the kernel in isolation.
    NEG_BIAS = -15.0  # safely inside Q4.11 range; exp(-15) ~ 3e-7 after softmax
    wij_raw = load_event("stage2_wij_post_mlp.npy", (N_MAX, N_MAX))
    wij_full = np.zeros((N_MAX, N_KV), dtype=np.float32)
    wij_full[:, :N_MAX] = wij_raw
    for j in range(N_MAX):
        if pad_mask[j]:
            wij_full[:, j] = NEG_BIAS
    wij_q = to_int16(wij_full)
    for h in range(N_HEADS):
        write_plio_text(os.path.join(args.data_dir, f"obj_wij_h{h}_L0.txt"), wij_q)

    # ---- cand0 inputs -------------------------------------------------------
    # Cand pipeline runs at Q6.9 (scale 512) so unnormalized cand_build sums
    # up to ~|19| fit. The cand kernel uses CAND_SCALE/CAND_ACC_SHIFT and the
    # cand weight headers are regenerated at the matching scale.
    CAND_SCALE = 1 << 9   # 512
    print("\n[cand0]")
    c_cand = load_event("stage3_layer0_candidates_embedded.npy", (T_DIM, E_DIM))
    c_cand_int = np.clip(np.round(c_cand * CAND_SCALE), -32768, 32767).astype(np.int16)
    write_plio_text(os.path.join(args.data_dir, "cand_c_in_L0.txt"), c_cand_int)

    # ---- cross0 inputs ------------------------------------------------------
    # Cross attention takes Q from post-obj-selfattn output and K=V from
    # post-cand-selfattn output (both at layer 0).
    print("\n[cross0]")
    x_cross = load_event("stage3_layer0_post_obj_selfattn.npy", (N_MAX, E_DIM))
    c_cross = load_event("stage3_layer0_post_cand_selfattn.npy", (T_DIM, E_DIM))
    write_plio_text(os.path.join(args.data_dir, "cross_x_in_L0.txt"), to_int16(x_cross))
    write_plio_text(os.path.join(args.data_dir, "cross_c_in_L0.txt"), to_int16(c_cross))

    # Save padding mask for the output checker so it can skip masked rows
    # the same way the PL compare<> helper does.
    np.save(os.path.join(args.data_dir, "padding_mask_event.npy"), pad_mask)

    print("\ndone.")


if __name__ == "__main__":
    main()