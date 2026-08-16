#!/usr/bin/env python
"""Pack the eval set as raw float32 words (FLOAT_DATAPATH golden input)."""
import sys, numpy as np, h5py, struct
def load(fn, n):
    with h5py.File(fn, "r") as f:
        e = np.nan_to_num(np.array(f['source']['e'])) / 1000.
        pt = np.nan_to_num(np.array(f['source']['pt'])) / 1000.
        with np.errstate(divide='ignore'):
            le = np.log(e);  le[~np.isfinite(le)] = 0
            lp = np.log(pt); lp[~np.isfinite(lp)] = 0
        phi = np.array(f['source']['phi']); eta = np.array(f['source']['eta'])
        X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], -1)
        X = X[(pt > 0).sum(1) >= 6][:n]
    out = np.zeros((X.shape[0], 12, 5), np.float32)
    k = min(12, X.shape[1]); out[:, :k, :] = X[:, :k, :]
    mask = (out[:, :, 0] == 0)
    return out, mask
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
j, m = load("/home/snehadri/repos/unsupervised-search/inputs/qcd_background.h5", n)
words = []
for ev in range(n):
    words += [int(w) for w in j[ev].reshape(-1).view(np.uint32)]
    words += [1 if x else 0 for x in m[ev]]
open("evalfloat_bkg.bin", "wb").write(struct.pack(f"{len(words)}I", *words))
print(f"packed {n} events (float32 words)")
