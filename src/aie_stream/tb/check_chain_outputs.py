#!/usr/bin/env python3
"""End-to-end check of the on-array stack: chain_x_out (x after cross L1) and chain_c_out
(c after candidate L1) against the PyTorch golden vectors, like check_attn_outputs --all-blocks."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_attn_outputs import parse_plio_text_float, DATA_SCALE, N_MAX, T_DIM, E_DIM

_DEFAULT_PHASE3 = "/home/snehadri/repos/unsupervised-search/phase3_export_retrained"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("outdir", help="Directory containing chain PLIO output text files")
ap.add_argument("nev", nargs="?", type=int, default=20, help="Number of events to check (default: 20)")
ap.add_argument("--phase3", default=_DEFAULT_PHASE3,
                help=f"Phase-3 export directory (default: {_DEFAULT_PHASE3})")
args = ap.parse_args()
outdir = args.outdir; nev = args.nev
tv = os.path.join(args.phase3, "test_vectors")
mask = np.load(f"{tv}/stage0_padding_mask.npy")[:nev]      # (nev, 12) True = padded
fails = 0
checks = [("x after cross L1", "chain_x_out.txt", "stage3_layer1_post_cross_attn.npy", N_MAX, True),
          ("c after cand L1",  "chain_c_out.txt", "stage3_layer1_post_cand_selfattn.npy", T_DIM, False)]
if os.path.exists(os.path.join(outdir, "chain_x0_out.txt")):   # the two-half graph
    checks.insert(0, ("x after cross L0", "chain_x0_out.txt", "stage3_layer0_post_cross_attn.npy", N_MAX, True))
for name, fname, gname, rows, masked in checks:
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
