#!/usr/bin/env python3
"""
slice_weights_for_aie.py
-------------------------
Reads the phase3_export .npy weight files (same ones used to generate
weights_rom.h for PL) and slices them into per-head / per-tile C headers
for the AIE attention kernels.

The bit pattern of ap_fixed<16,5> is identical to int16 with 11 fractional
bits (DATA_FRAC_BITS=11), so we use the same quantization.

Usage:
    python slice_weights_for_aie.py \
        --export_dir /home/snehadri/repos/unsupervised-search/phase3_export \
        --out_dir    /home/snehadri/repos/aie-unsupervised-search/aie_attn/src/kernels/weights \
        --layer 0

    Repeat with --layer 1 for the second ABC layer.
"""

import argparse
import os
import numpy as np
import glob

# ============================================================
# Constants (must match attn_aie_types.h)
# ============================================================
E_DIM = 16
D_HEAD = 4
N_HEADS = 4
T_DIM = 3
DATA_FRAC_BITS = 11
WEIGHT_FRAC_BITS = 11  # ap_fixed<16,5> has 11 frac bits, same as data

def to_fixed16(arr, frac_bits=11):
    """Convert float array to int16 fixed-point."""
    scale = 2 ** frac_bits
    return np.clip(np.round(arr * scale), -32768, 32767).astype(np.int16)

def format_array(name, arr, per_line=8):
    """Format int16 array as C initializer."""
    flat = arr.flatten()
    lines = [f"alignas(16) static const int16 {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        vals = ", ".join(f"{int(v):6d}" for v in chunk)
        comma = "," if i + per_line < len(flat) else ""
        lines.append(f"    {vals}{comma}")
    lines.append("};")
    return "\n".join(lines)

def load_npy(export_dir, prefix, name):
    """Load a .npy file from phase3_export."""
    path = os.path.join(export_dir, f"{prefix}{name}.npy")
    if not os.path.exists(path):
        # try without dots (some exports use _ instead of .)
        alt = path.replace(".", "_").replace("_npy", ".npy")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Cannot find {path}")
    arr = np.load(path)
    print(f"  Loaded {path}: shape={arr.shape}")
    return arr

def slice_per_head(full_weight, head_idx):
    """
    Slice per-head projection from full in_proj_weight.

    PyTorch MHA in_proj_weight is [3*E, E] (Q, K, V stacked).
    For AIE kernel: W_aie[E][D_HEAD] = transpose of rows [h*d:(h+1)*d]
    from each Q/K/V block.

    The AIE kernel computes X @ W + b (not X @ W^T + b),
    so W_aie = PyTorch_W_block[h*d:(h+1)*d, :].T → [E, D_HEAD]
    """
    E = E_DIM
    d = D_HEAD
    h = head_idx

    Wq_block = full_weight[0:E, :]      # [E, E]
    Wk_block = full_weight[E:2*E, :]
    Wv_block = full_weight[2*E:3*E, :]

    Wq_h = Wq_block[h*d:(h+1)*d, :].T   # [E, d]
    Wk_h = Wk_block[h*d:(h+1)*d, :].T
    Wv_h = Wv_block[h*d:(h+1)*d, :].T

    return Wq_h, Wk_h, Wv_h

def slice_bias_per_head(full_bias, head_idx):
    """Slice per-head bias from [3*E] in_proj_bias."""
    E = E_DIM
    d = D_HEAD
    h = head_idx

    bq = full_bias[h*d:(h+1)*d]
    bk = full_bias[E + h*d:E + (h+1)*d]
    bv = full_bias[2*E + h*d:2*E + (h+1)*d]
    return bq, bk, bv

def slice_bias_kv_per_head(bias_k_full, bias_v_full, head_idx):
    """Slice per-head bias_kv from [E] → [D_HEAD]."""
    d = D_HEAD
    h = head_idx
    return bias_k_full[h*d:(h+1)*d], bias_v_full[h*d:(h+1)*d]

def write_head_header(out_dir, attn_type, layer, head_idx, weights_dict):
    """Write one per-head weight header."""
    fname = f"{attn_type}_head{head_idx}_weights_L{layer}.h"
    guard = f"{attn_type.upper()}_HEAD{head_idx}_WEIGHTS_L{layer}_H"
    path = os.path.join(out_dir, fname)

    with open(path, 'w') as f:
        f.write(f"// Auto-generated from phase3_export .npy files\n")
        f.write(f"// Attention type: {attn_type}, layer: {layer}, head: {head_idx}\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")

        for name, arr in weights_dict.items():
            f.write(format_array(name, arr) + "\n\n")

        f.write(f"#endif // {guard}\n")

    print(f"  Wrote {path}")

def write_post_header(out_dir, attn_type, layer, weights_dict):
    """Write post-attention weight header (shared across heads)."""
    fname = f"{attn_type}_post_weights_L{layer}.h"
    guard = f"{attn_type.upper()}_POST_WEIGHTS_L{layer}_H"
    path = os.path.join(out_dir, fname)

    with open(path, 'w') as f:
        f.write(f"// Auto-generated from phase3_export .npy files\n")
        f.write(f"// Post-attention weights: {attn_type}, layer: {layer}\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")

        for name, arr in weights_dict.items():
            f.write(format_array(name, arr) + "\n\n")

        f.write(f"#endif // {guard}\n")

    print(f"  Wrote {path}")

def process_attn_block(export_dir, out_dir, attn_type, layer):
    """Process one attention block type (obj/cand/cross) for one layer."""
    # Map attention type to phase3_export prefix
    prefix_map = {
        "obj":   f"obj_blocks_{layer}_",
        "cand":  f"cand_blocks_{layer}_",
        "cross": f"cross_blocks_{layer}_",
    }
    prefix = prefix_map[attn_type]

    print(f"\nProcessing {attn_type} attention, layer {layer}")
    print(f"  Prefix: {prefix}")

    # Load full projection weights
    in_proj_w = load_npy(export_dir, prefix, "attn_in_proj_weight")  # [3E, E]
    in_proj_b = load_npy(export_dir, prefix, "attn_in_proj_bias")    # [3E]

    # Load bias_kv
    bias_k_full = load_npy(export_dir, prefix, "attn_bias_k").squeeze()  # [E]
    bias_v_full = load_npy(export_dir, prefix, "attn_bias_v").squeeze()  # [E]

    # Per-head weights
    for h in range(N_HEADS):
        Wq_h, Wk_h, Wv_h = slice_per_head(in_proj_w, h)
        bq_h, bk_h, bv_h = slice_bias_per_head(in_proj_b, h)
        bk_row_h, bv_row_h = slice_bias_kv_per_head(bias_k_full, bias_v_full, h)

        # Determine weight name prefix for this attention type
        w_prefix = "" if attn_type == "obj" else f"{attn_type}_"

        weights = {
            f"{w_prefix}Wq": to_fixed16(Wq_h),
            f"{w_prefix}bq": to_fixed16(bq_h),
            f"{w_prefix}Wk": to_fixed16(Wk_h),
            f"{w_prefix}bk": to_fixed16(bk_h),
            f"{w_prefix}Wv": to_fixed16(Wv_h),
            f"{w_prefix}bv": to_fixed16(bv_h),
            f"{w_prefix}bias_k_row": to_fixed16(bk_row_h),
            f"{w_prefix}bias_v_row": to_fixed16(bv_row_h),
        }

        write_head_header(out_dir, attn_type, layer, h, weights)

    # Post-attention weights (shared across heads)
    # Output projection
    out_proj_w = load_npy(export_dir, prefix, "attn_out_proj_weight")  # [E, E]
    out_proj_b = load_npy(export_dir, prefix, "attn_out_proj_bias")    # [E]

    # Post-attention LayerNorm
    post_attn_ln_w = load_npy(export_dir, prefix, "post_attn_norm_weight")  # [E]
    post_attn_ln_b = load_npy(export_dir, prefix, "post_attn_norm_bias")    # [E]

    # FFN layers: Sequential with stride-of-3 (Linear, LayerNorm, ReLU)
    post_weights = {
        "Wout": to_fixed16(out_proj_w.T),  # transpose for AIE (X @ W, not X @ W^T)
        "bout": to_fixed16(out_proj_b),
        "post_attn_ln_gamma": to_fixed16(post_attn_ln_w),
        "post_attn_ln_beta":  to_fixed16(post_attn_ln_b),
    }

    # FFN layers
    for ffn_idx in range(3):
        seq_idx = ffn_idx * 3  # Linear at 0, 3, 6

        ffn_w = load_npy(export_dir, prefix, f"ffwd_{seq_idx}_weight")  # [out, in]
        ffn_b = load_npy(export_dir, prefix, f"ffwd_{seq_idx}_bias")

        post_weights[f"ffn_W{ffn_idx}"] = to_fixed16(ffn_w.T)  # transpose
        post_weights[f"ffn_b{ffn_idx}"] = to_fixed16(ffn_b)

        # LayerNorm at seq_idx+1 (if it exists)
        ln_w_path = os.path.join(export_dir, f"{prefix}ffwd_{seq_idx+1}_weight.npy")
        if os.path.exists(ln_w_path):
            ln_w = np.load(ln_w_path)
            ln_b = np.load(os.path.join(export_dir, f"{prefix}ffwd_{seq_idx+1}_bias.npy"))
            post_weights[f"ffn_ln_gamma{ffn_idx}"] = to_fixed16(ln_w)
            post_weights[f"ffn_ln_beta{ffn_idx}"]  = to_fixed16(ln_b)

    # Post-FFN LayerNorm
    pffn_path = os.path.join(export_dir, f"{prefix}post_ffwd_norm_weight.npy")
    if os.path.exists(pffn_path):
        pffn_w = np.load(pffn_path)
        pffn_b = np.load(os.path.join(export_dir, f"{prefix}post_ffwd_norm_bias.npy"))
        post_weights["post_ffn_ln_gamma"] = to_fixed16(pffn_w)
        post_weights["post_ffn_ln_beta"]  = to_fixed16(pffn_b)

    write_post_header(out_dir, attn_type, layer, post_weights)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", required=True,
                        help="Path to phase3_export/ directory with .npy files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for AIE weight headers")
    parser.add_argument("--layer", type=int, required=True,
                        help="Attention layer index (0 or 1)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # List available .npy files for debugging
    npy_files = sorted(glob.glob(os.path.join(args.export_dir, "*.npy")))
    print(f"Found {len(npy_files)} .npy files in {args.export_dir}")
    if len(npy_files) < 5:
        print("WARNING: Very few .npy files found. Check --export_dir path.")
        for f in npy_files:
            print(f"  {os.path.basename(f)}")

    # Process all three attention types for this layer
    for attn_type in ["obj", "cand", "cross"]:
        process_attn_block(args.export_dir, args.out_dir, attn_type, args.layer)

    print(f"\nDone! Generated headers in {args.out_dir}")
    print(f"Per-head headers: 4 heads × 3 attn types = 12 files")
    print(f"Post-attn headers: 3 files")
    print(f"\nNext: #include the appropriate header in each AIE kernel .cc file")

if __name__ == "__main__":
    main()