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
                    help="Event index to start at (default: 0)")
    ap.add_argument("--num-events", type=int, default=1,
                    help="Number of events to concatenate into each PLIO file (default: 1)")
    ap.add_argument("--data-dir", default="./data",
                    help="Where to write PLIO text files (default: ./data)")
    args = ap.parse_args()

    tv = os.path.join(args.phase3, "test_vectors")
    if not os.path.isdir(tv):
        sys.exit(f"test_vectors directory not found: {tv}")

    ev0 = args.event
    nev = args.num_events
    print(f"generating PLIO inputs for events [{ev0}..{ev0+nev-1}]  (N={nev})")
    print(f"  test_vectors: {tv}")
    print(f"  output dir:   {args.data_dir}")

    # Lazily-loaded full arrays
    arrays = {}
    def full(name):
        if name not in arrays:
            arrays[name] = np.load(os.path.join(tv, name))
        return arrays[name]

    def event_slice(name, expected_shape):
        """Return shape (nev,)+expected_shape slice for the requested event range."""
        arr = full(name)
        if arr.shape[0] < ev0 + nev:
            sys.exit(f"{name}: only {arr.shape[0]} events, "
                     f"requested [{ev0}..{ev0+nev-1}]")
        out = arr[ev0:ev0+nev]
        if out.shape[1:] != expected_shape:
            sys.exit(f"{name}: expected shape (N,)+{expected_shape}, got {out.shape}")
        return out

    pad_masks = full("stage0_padding_mask.npy")[ev0:ev0+nev].astype(bool)  # (nev, N_MAX)
    if pad_masks.shape != (nev, N_MAX):
        sys.exit(f"padding_mask: expected shape ({nev},{N_MAX}), got {pad_masks.shape}")

    # Helper: concatenate int16 arrays from N events into one flat array.
    def concat_events(per_event_int16):
        # per_event_int16: list of length nev, each a flat int16 array
        return np.concatenate([a.reshape(-1) for a in per_event_int16])

    NEG_BIAS = -15.0  # for masked-out keys in wij
    CAND_SCALE = 1 << 9   # cand Q6.9

    # ---- obj0 inputs --------------------------------------------------------
    print("\n[obj0]")
    x_obj_per = event_slice("stage1_post_embedding.npy", (N_MAX, E_DIM))     # (nev,12,16)
    x_obj_q   = [to_int16(x_obj_per[i]) for i in range(nev)]
    write_plio_text(os.path.join(args.data_dir, "obj_x_in_L0.txt"),
                    concat_events(x_obj_q))

    wij_raw_per = event_slice("stage2_wij_post_mlp.npy", (N_MAX, N_MAX))     # (nev,12,12)
    wij_q_per = []
    for i in range(nev):
        wij_full = np.zeros((N_MAX, N_KV), dtype=np.float32)
        wij_full[:, :N_MAX] = wij_raw_per[i]
        for j in range(N_MAX):
            if pad_masks[i, j]:
                wij_full[:, j] = NEG_BIAS
        wij_q_per.append(to_int16(wij_full))
    for h in range(N_HEADS):
        write_plio_text(os.path.join(args.data_dir, f"obj_wij_h{h}_L0.txt"),
                        concat_events(wij_q_per))

    # ---- cand0 inputs -------------------------------------------------------
    print("\n[cand0]")
    c_cand_per = event_slice("stage3_layer0_candidates_embedded.npy", (T_DIM, E_DIM))
    c_cand_int_per = [np.clip(np.round(c_cand_per[i] * CAND_SCALE),
                              -32768, 32767).astype(np.int16) for i in range(nev)]
    write_plio_text(os.path.join(args.data_dir, "cand_c_in_L0.txt"),
                    concat_events(c_cand_int_per))

    # ---- cross0 inputs ------------------------------------------------------
    print("\n[cross0]")
    x_cross_per = event_slice("stage3_layer0_post_obj_selfattn.npy", (N_MAX, E_DIM))
    c_cross_per = event_slice("stage3_layer0_post_cand_selfattn.npy", (T_DIM, E_DIM))
    write_plio_text(os.path.join(args.data_dir, "cross_x_in_L0.txt"),
                    concat_events([to_int16(x_cross_per[i]) for i in range(nev)]))
    write_plio_text(os.path.join(args.data_dir, "cross_c_in_L0.txt"),
                    concat_events([to_int16(c_cross_per[i]) for i in range(nev)]))

    # Save padding masks for all events so the output checker can skip masked rows
    np.save(os.path.join(args.data_dir, "padding_mask_event.npy"), pad_masks)
    # Save event range so checker knows what range to validate
    with open(os.path.join(args.data_dir, "event_range.txt"), "w") as f:
        f.write(f"{ev0} {nev}\n")

    print(f"\ndone. wrote PLIO files for {nev} events.")


if __name__ == "__main__":
    main()