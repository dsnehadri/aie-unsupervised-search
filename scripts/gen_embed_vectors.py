#!/usr/bin/env python3
"""Write the PLIO input file for the embedding graph and the float golden.

Input rows are the raw jet features at Q6.9, four int16 per 64-bit PLIO word.
"""
import numpy as np, sys, os
T = "/home/snehadri/repos/unsupervised-search/phase3_export_retrained/test_vectors"
D = sys.argv[1] if len(sys.argv) > 1 else "src/aie_stream/tb/data"
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 20
FRAC = 9

raw = np.load(f"{T}/stage0_input_raw.npy")[:NEV]          # (NEV, 12, 5)
gold = np.load(f"{T}/stage1_post_embedding.npy")[:NEV]    # (NEV, 12, 16)
q = np.clip(np.rint(raw * (1 << FRAC)), -32768, 32767).astype(np.int16)

os.makedirs(D, exist_ok=True)
with open(f"{D}/embed_jets_in.txt", "w") as f:
    flat = q.reshape(NEV, -1)
    for ev in flat:
        for i in range(0, ev.size, 4):
            f.write(" ".join(str(int(v)) for v in ev[i:i+4]) + "\n")
np.save(f"{D}/embed_golden.npy", gold)
print(f"wrote {D}/embed_jets_in.txt ({NEV} events, {q[0].size} int16 each) and embed_golden.npy")
print(f"  raw feature range +/-{np.abs(raw).max():.2f}, golden range +/-{np.abs(gold).max():.3f}")
