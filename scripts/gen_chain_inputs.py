#!/usr/bin/env python3
"""PLIO inputs for the whole-chain graph: embed_jets_in.txt and mask_in.txt.

The chain takes only these two files; every other array input is produced on
the array. gen_attn_inputs.py writes the per-block files instead, and the chain
pair was previously made by hand, which is how it went stale and left the
aiesimulator waiting on inputs that no longer matched the graph.

  embed_jets_in.txt  12 jets x 5 raw features per event, Q6.9, four int16 a line
  mask_in.txt        one E_DIM window per event: 1 marks a padded jet, 0 a real
                     one, zero-filled past N_MAX

usage: gen_chain_inputs.py [data_dir] [n_events]
"""
import os
import sys

import numpy as np
from paths import DATA

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/aie_stream/tb/data")
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_MAX, E_DIM, FRAC = 12, 16, 9
EMBED_IN_WORDS = 64     # the graph's jets window: 12 x 5 features, then padding

raw = np.load(f"{DATA}/stage0_input_raw.npy")[:NEV]        # (NEV, 12, 5) float
pad = np.load(f"{DATA}/stage0_padding_mask.npy")[:NEV]     # (NEV, 12) bool, True = padded
assert raw.shape[0] == NEV and pad.shape[0] == NEV, "not enough events in the test vectors"

def write_plio(path, values):
    """Four int16 to a 64-bit PLIO word, one word a line."""
    flat = np.asarray(values, dtype=np.int16).ravel()
    if flat.size % 4:
        flat = np.concatenate([flat, np.zeros(-flat.size % 4, dtype=np.int16)])
    with open(path, "w") as f:
        for i in range(0, flat.size, 4):
            f.write(" ".join(str(int(v)) for v in flat[i:i + 4]) + "\n")
    return flat.size

q = np.clip(np.rint(raw * (1 << FRAC)), -32768, 32767).astype(np.int16)
jets = np.zeros((NEV, EMBED_IN_WORDS), dtype=np.int16)      # pad each event to the window
jets[:, :N_MAX * raw.shape[2]] = q.reshape(NEV, -1)
n = write_plio(os.path.join(OUT, "embed_jets_in.txt"), jets)

masks = np.zeros((NEV, E_DIM), dtype=np.int16)
masks[:, :N_MAX] = pad[:, :N_MAX].astype(np.int16)
m = write_plio(os.path.join(OUT, "mask_in.txt"), masks)
print(f"wrote {OUT}/embed_jets_in.txt ({n} int16) and mask_in.txt ({m} int16), {NEV} events")
print(f"jets per event: {[int(N_MAX - pad[i, :N_MAX].sum()) for i in range(min(NEV, 8))]} ...")
