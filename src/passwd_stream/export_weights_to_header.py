#!/usr/bin/env python3
"""
convert .npy weights to a C header for BRAM ROM.
Emits plain float arrays + init functions that copy float→ap_fixed.

Usage:
    python export_weights_to_header.py \
        --weight_dir /home/snehadri/repos/unsupervised-search/phase3_export/weights/ \
        --output     /home/snehadri/repos/aie-unsupervised-search/passwd_stream_source/weights_rom.h
"""

import argparse
import numpy as np
import os


def to_c(arr):
    return np.array2string(arr, separator=",", threshold=np.inf,
                           max_line_width=np.inf,
                           formatter={"float_kind": lambda x: f"{x:.10f}"}
                           ).replace("[", "{").replace("]", "}")


def shape_str(arr):
    return "".join(f"[{d}]" for d in arr.shape)


def emit_float_array(f, name, arr):
    f.write(f"static const float {name}{shape_str(arr)} = {to_c(arr)};\n")


def emit_copy(f, dst, src, arr, indent="    "):
    if arr.ndim == 1:
        f.write(f"{indent}for(int i=0;i<{arr.shape[0]};i++) {dst}[i]={src}[i];\n")
    elif arr.ndim == 2:
        f.write(f"{indent}for(int i=0;i<{arr.shape[0]};i++) for(int j=0;j<{arr.shape[1]};j++) {dst}[i][j]={src}[i][j];\n")
    elif arr.ndim == 3:
        f.write(f"{indent}for(int i=0;i<{arr.shape[0]};i++) for(int j=0;j<{arr.shape[1]};j++) for(int k=0;k<{arr.shape[2]};k++) {dst}[i][j][k]={src}[i][j][k];\n")


def load(weight_dir, fname):
    return np.load(os.path.join(weight_dir, fname))


# ============================================================================
# DNNBlockWeights<IN, HIDDEN, OUT, N_MID>
#   first_w[HIDDEN][IN], first_b[HIDDEN], first_ln_g[HIDDEN], first_ln_b[HIDDEN]
#   mid_w[N_MID][HIDDEN][HIDDEN], mid_b[N_MID][HIDDEN],
#   mid_ln_g[N_MID][HIDDEN], mid_ln_b[N_MID][HIDDEN]
#   last_w[OUT][HIDDEN], last_b[OUT]
#
# Layer index mapping (stride-3 in PyTorch Sequential):
#   first = idx 0 (Linear), LN = idx 1, ReLU = idx 2
#   mid[0] = idx 3, LN = idx 4, ReLU = idx 5
#   mid[1] = idx 6, LN = idx 7, ReLU = idx 8  (if N_MID >= 2)
#   last = idx 3*(N_MID+1)
# ============================================================================

def dnn_fields(prefix, weight_dir, n_mid):
    """Return list of (field_name, numpy_array) matching DNNBlockWeights layout."""
    fields = []

    # first layer (idx 0 = Linear, idx 1 = LN)
    fields.append(("first_w",    load(weight_dir, f"{prefix}0_weight.npy")))
    fields.append(("first_b",    load(weight_dir, f"{prefix}0_bias.npy")))
    fields.append(("first_ln_g", load(weight_dir, f"{prefix}1_weight.npy")))
    fields.append(("first_ln_b", load(weight_dir, f"{prefix}1_bias.npy")))

    # mid layers: stack into [N_MID][HIDDEN][HIDDEN] etc.
    if n_mid > 0:
        mid_linear_idx = [3 + 3 * i for i in range(n_mid)]  # 3, 6, 9, ...
        mid_ln_idx = [4 + 3 * i for i in range(n_mid)]       # 4, 7, 10, ...

        fields.append(("mid_w",    np.stack([load(weight_dir, f"{prefix}{i}_weight.npy") for i in mid_linear_idx])))
        fields.append(("mid_b",    np.stack([load(weight_dir, f"{prefix}{i}_bias.npy") for i in mid_linear_idx])))
        fields.append(("mid_ln_g", np.stack([load(weight_dir, f"{prefix}{i}_weight.npy") for i in mid_ln_idx])))
        fields.append(("mid_ln_b", np.stack([load(weight_dir, f"{prefix}{i}_bias.npy") for i in mid_ln_idx])))

    # last layer (bare, no LN)
    last_idx = 3 * (n_mid + 1)
    fields.append(("last_w", load(weight_dir, f"{prefix}{last_idx}_weight.npy")))
    fields.append(("last_b", load(weight_dir, f"{prefix}{last_idx}_bias.npy")))

    return fields


# ============================================================================
# AttnWeights
#   Wq[16][16], bq[16], Wk[16][16], bk[16], Wv[16][16], bv[16]
#   bias_k[16], bias_v[16], Wo[16][16], bo[16]
#   attn_ln_g[16], attn_ln_b[16]
#   ffn_w[3][16][16], ffn_b[3][16], ffn_ln_g[3][16], ffn_ln_b[3][16]
#   post_ffn_g[16], post_ffn_b[16]
# ============================================================================

def attn_fields(block_name, weight_dir, e_dim=16):
    p = os.path.join(weight_dir, block_name + "_")

    # Split combined in_proj into Q/K/V
    in_proj_w = np.load(p + "attn_in_proj_weight.npy")  # [3*E, E]
    in_proj_b = np.load(p + "attn_in_proj_bias.npy")    # [3*E]

    fields = [
        ("Wq", in_proj_w[0*e_dim:1*e_dim]),
        ("bq", in_proj_b[0*e_dim:1*e_dim]),
        ("Wk", in_proj_w[1*e_dim:2*e_dim]),
        ("bk", in_proj_b[1*e_dim:2*e_dim]),
        ("Wv", in_proj_w[2*e_dim:3*e_dim]),
        ("bv", in_proj_b[2*e_dim:3*e_dim]),
        ("bias_k",    np.load(p + "attn_bias_k.npy").flatten()),
        ("bias_v",    np.load(p + "attn_bias_v.npy").flatten()),
        ("Wo",        np.load(p + "attn_out_proj_weight.npy")),
        ("bo",        np.load(p + "attn_out_proj_bias.npy")),
        ("attn_ln_g", np.load(p + "post_attn_norm_weight.npy")),
        ("attn_ln_b", np.load(p + "post_attn_norm_bias.npy")),
    ]

    # FFN: stride-3 indices (0=Linear, 1=LN, 2=ReLU, 3=Linear, ...)
    ffn_li = [0, 3, 6]
    ffn_ln = [1, 4, 7]
    fields += [
        ("ffn_w",    np.stack([np.load(p + f"ffwd_{i}_weight.npy") for i in ffn_li])),
        ("ffn_b",    np.stack([np.load(p + f"ffwd_{i}_bias.npy") for i in ffn_li])),
        ("ffn_ln_g", np.stack([np.load(p + f"ffwd_{i}_weight.npy") for i in ffn_ln])),
        ("ffn_ln_b", np.stack([np.load(p + f"ffwd_{i}_bias.npy") for i in ffn_ln])),
        ("post_ffn_g", np.load(p + "post_ffwd_norm_weight.npy")),
        ("post_ffn_b", np.load(p + "post_ffwd_norm_bias.npy")),
    ]
    return fields


# ============================================================================
# AEEncoderWeights / AEDecoderWeights
#   w0[D1][D0], b0[D1], ln0_g[D1], ln0_b[D1]
#   w1[D2][D1], b1[D2], ln1_g[D2], ln1_b[D2]
#   w2[D3][D2], b2[D3], ln2_g[D3], ln2_b[D3]
#   w3[D4][D3], b3[D4]    (bare, no LN)
# ============================================================================

def ae_fields(prefix, weight_dir):
    linear_idx = [0, 3, 6, 9]
    ln_idx = [1, 4, 7]
    fields = []
    for i, li in enumerate(linear_idx):
        fields.append((f"w{i}", load(weight_dir, f"{prefix}{li}_weight.npy")))
        fields.append((f"b{i}", load(weight_dir, f"{prefix}{li}_bias.npy")))
        if i < len(ln_idx):
            fields.append((f"ln{i}_g", load(weight_dir, f"{prefix}{ln_idx[i]}_weight.npy")))
            fields.append((f"ln{i}_b", load(weight_dir, f"{prefix}{ln_idx[i]}_bias.npy")))
    return fields


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    wd = args.weight_dir

    # Collect all structs: (tag, c_type, rom_name, fields)
    structs = []

    # Embed: DNNBlockWeights<5,16,16,1> — N_MID=1
    structs.append(("EMBED", "EmbedWeights", "ROM_EMBED_W",
                    dnn_fields("embed_net_", wd, n_mid=1)))

    # MLP: DNNBlockWeights<3,16,1,2> — N_MID=2
    structs.append(("MLP", "MLPWeights", "ROM_MLP_W",
                    dnn_fields("mlp_net_", wd, n_mid=2)))

    # 6 attention blocks
    for block, tag, rom in [("obj_blocks_0",   "OBJ0",   "ROM_OBJ0_W"),
                            ("cand_blocks_0",  "CAND0",  "ROM_CAND0_W"),
                            ("cross_blocks_0", "CROSS0", "ROM_CROSS0_W"),
                            ("obj_blocks_1",   "OBJ1",   "ROM_OBJ1_W"),
                            ("cand_blocks_1",  "CAND1",  "ROM_CAND1_W"),
                            ("cross_blocks_1", "CROSS1", "ROM_CROSS1_W")]:
        structs.append((tag, "AttnWeights", rom, attn_fields(block, wd)))

    # Autoencoders
    structs.append(("AE_ENC", "AEEncoderWeights", "ROM_AE_ENC_W",
                    ae_fields("ae_in_net_", wd)))
    structs.append(("AE_DEC", "AEDecoderWeights", "ROM_AE_DEC_W",
                    ae_fields("ae_out_net_", wd)))

    with open(args.output, "w") as f:
        f.write("#ifndef WEIGHTS_ROM_H\n#define WEIGHTS_ROM_H\n\n")
        f.write("// Auto-generated — do not edit\n\n")
        f.write('#include "../attn_block_pl/attn_block_types.h"\n')
        f.write('#include "../attn_block_pl/attn_helpers.h"\n')
        f.write('#include "../embed_ffn/embed_ffn.h"\n')
        f.write('#include "../pairwise_mlp/pairwise_mlp.h"\n')
        f.write('#include "../autoencoder/autoencoder.h"\n\n')

        # Part 1: float arrays
        for tag, c_type, rom_name, fields in structs:
            f.write(f"// ---- {tag} ----\n")
            for field, arr in fields:
                emit_float_array(f, f"{rom_name}_{field}", arr)
            f.write("\n")

        # Part 2: init functions
        for tag, c_type, rom_name, fields in structs:
            f.write(f"inline void init_{rom_name.lower()}({c_type} &w) {{\n")
            for field, arr in fields:
                emit_copy(f, f"w.{field}", f"{rom_name}_{field}", arr)
            f.write("}\n\n")

        # Part 3: convenience init_all_weights
        f.write("inline void init_all_weights(\n")
        f.write("    EmbedWeights &embed_w, MLPWeights &mlp_w,\n")
        f.write("    AttnWeights &obj0_w, AttnWeights &cand0_w, AttnWeights &cross0_w,\n")
        f.write("    AttnWeights &obj1_w, AttnWeights &cand1_w, AttnWeights &cross1_w,\n")
        f.write("    AEEncoderWeights &ae_enc_w, AEDecoderWeights &ae_dec_w\n")
        f.write(") {\n")
        f.write("    init_rom_embed_w(embed_w);\n")
        f.write("    init_rom_mlp_w(mlp_w);\n")
        f.write("    init_rom_obj0_w(obj0_w);\n")
        f.write("    init_rom_cand0_w(cand0_w);\n")
        f.write("    init_rom_cross0_w(cross0_w);\n")
        f.write("    init_rom_obj1_w(obj1_w);\n")
        f.write("    init_rom_cand1_w(cand1_w);\n")
        f.write("    init_rom_cross1_w(cross1_w);\n")
        f.write("    init_rom_ae_enc_w(ae_enc_w);\n")
        f.write("    init_rom_ae_dec_w(ae_dec_w);\n")
        f.write("}\n\n")

        f.write("#endif\n")

    print(f"Generated {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")


if __name__ == "__main__":
    main()