#!/usr/bin/env python3
"""End-to-end check of the on-array stack: chain_x_out (x after cross L1) and chain_c_out
(c after candidate L1) against the PyTorch golden vectors, like check_attn_outputs --all-blocks."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_attn_outputs import parse_plio_text_float, DATA_SCALE, N_MAX, T_DIM, E_DIM
outdir = sys.argv[1]; nev = int(sys.argv[2]) if len(sys.argv) > 2 else 20
tv = "/home/snehadri/repos/unsupervised-search/phase3_export_retrained/test_vectors"
mask = np.load(f"{tv}/stage0_padding_mask.npy")[:nev]      # (nev, 12) True = padded
fails = 0
for name, fname, gname, rows, masked in (("x after cross L1", "chain_x_out.txt", "stage3_layer1_post_cross_attn.npy", N_MAX, True),
                                          ("c after cand L1",  "chain_c_out.txt", "stage3_layer1_post_cand_selfattn.npy", T_DIM, False)):
    d = parse_plio_text_float(os.path.join(outdir, fname)) / DATA_SCALE
    gold = np.load(f"{tv}/{gname}")[:nev]
    per = rows * E_DIM; errs = []
    for i in range(min(nev, len(d) // per)):
        got = d[i * per:(i + 1) * per].reshape(rows, E_DIM); g = gold[i].reshape(rows, E_DIM)
        e = np.abs(got - g)
        if masked: e[mask[i]] = 0
        errs.append(e.max())
    errs = np.array(errs)
    print(f"{name:18s}: events {len(errs)}, max err {errs.max():.4f}, mean {errs.mean():.4f}, per event {np.round(errs, 3).tolist()}")
    fails += int((errs > 0.5).sum())
print("CHAIN_CHECK", "PASSED" if fails == 0 else f"FAILED ({fails})")
