#!/usr/bin/env python
"""Baseline anomaly metrics vs the AE reconstruction loss (AUC table).

Baselines, from trivial to classical:
  - H_T: scalar sum of jet pT
  - N_jets: jet multiplicity
  - m_avg(min-asym): brute-force minimum mass-asymmetry partition of the
    jets into two groups; score = average group mass (the classical
    combinatorial approach the ABC layers replace)
AE loss comes from the cached per-event losses (ae_losses.npz).

Run:  cd ~/repos/unsupervised-search && \
      ~/miniconda3/envs/unsupervised_search/bin/python \
      ~/repos/aie-unsupervised-search/scripts/anomaly_baselines.py
"""
import os, sys, json
import numpy as np

CACHE = "/home/snehadri/aie_scratch_save_20260810/anomaly_baselines.npz"
AE_CACHE = "/home/snehadri/aie_scratch_save_20260810/ae_losses.npz"
OUT_JSON = "/home/snehadri/aie_scratch_save_20260810/anomaly_baseline_aucs.json"
N_BKG = 200000

SAMPLES = ["qcd_background", "gluino_rpv_6j", "gluino_rpv_10j",
           "stop_rpv_12j", "squark_rpv_8j_WZH_2000",
           "squark_rpv_8j_2000"]

if not os.path.isfile(CACHE):
    import itertools, torch, h5py
    sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
    from model_blocks import x_to_p4

    def load(fn, n=None):
        with h5py.File(fn, "r") as f:
            e = np.nan_to_num(np.array(f['source']['e'])) / 1000.
            pt = np.nan_to_num(np.array(f['source']['pt'])) / 1000.
            with np.errstate(divide='ignore'):
                le = np.log(e); le[~np.isfinite(le)] = 0
                lp = np.log(pt); lp[~np.isfinite(lp)] = 0
            phi = np.array(f['source']['phi']); eta = np.array(f['source']['eta'])
            X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], -1)
            X = X[(pt > 0).sum(1) >= 6]
        if n: X = X[:n]
        return torch.tensor(X, dtype=torch.float32)

    def min_asym_mavg(jp4, njet):
        N = jp4.shape[0]
        out = np.full(N, np.nan)
        for n in np.unique(njet):
            n = int(n)
            if n < 2: continue
            combos = []
            for r in range(0, n):
                for extra in itertools.combinations(range(1, n), r):
                    m = np.zeros(n, bool); m[0] = True
                    for e_ in extra: m[e_] = True
                    combos.append(m)
            masks = np.array(combos)
            sel = np.where(njet == n)[0]
            if len(sel) == 0: continue
            mf = masks.astype(float)
            def m(p):
                m2 = p[..., 0]**2 - p[..., 1]**2 - p[..., 2]**2 - p[..., 3]**2
                return np.sqrt(np.clip(m2, 0, None))
            # chunk over events: the (E,P,4) intermediate is ~13 GB at 200k
            CH = 2000
            for c0 in range(0, len(sel), CH):
                idx = sel[c0:c0 + CH]
                jets = jp4[idx][:, :n, :]
                A = np.einsum('pn,enf->epf', mf, jets)
                B = jets.sum(1)[:, None, :] - A
                mA, mB = m(A), m(B)
                valid = (mA + mB) > 0
                asym = np.where(valid, np.abs(mA - mB) / (mA + mB + 1e-9), np.inf)
                best = asym.argmin(1)
                ei = np.arange(len(idx))
                out[idx] = 0.5 * (mA[ei, best] + mB[ei, best])
        return out

    arrs = {}
    for s in SAMPLES:
        X = load(f"inputs/{s}.h5", N_BKG if s == "qcd_background" else None)
        jp4 = x_to_p4(X).numpy()                       # (N,12,4) GeV
        pt = np.sqrt(jp4[:, :, 1]**2 + jp4[:, :, 2]**2)
        pt[jp4[:, :, 0] == 0] = 0
        njet = (jp4[:, :, 0] > 0).sum(1)
        arrs[f"{s}_ht"] = pt.sum(1)
        arrs[f"{s}_njet"] = njet.astype(float)
        arrs[f"{s}_mavg"] = min_asym_mavg(jp4, njet)
        print(s, "done", len(njet))
    np.savez(CACHE, **arrs)
    print("cached", CACHE)

d = np.load(CACHE)
ae = np.load(AE_CACHE)

def auc(bkg, sig):
    ok_b, ok_s = np.isfinite(bkg), np.isfinite(sig)
    a = np.concatenate([bkg[ok_b], sig[ok_s]])
    r = a.argsort().argsort()
    nb, ns = ok_b.sum(), ok_s.sum()
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)

metrics = [("H_T", "ht"), ("N_jets", "njet"), ("m_avg (min-asym)", "mavg")]
sigs = SAMPLES[1:]
print(f"\n{'metric':22s}" + "".join(f"{s[:18]:>20s}" for s in sigs))
table = {}
for label, key in metrics:
    row = [auc(d[f"qcd_background_{key}"], d[f"{s}_{key}"]) for s in sigs]
    table[label] = row
    print(f"{label:22s}" + "".join(f"{v:20.3f}" for v in row))
row = [auc(ae["qcd_background"], ae[s]) for s in sigs]
table["AE loss (this work)"] = row
print(f"{'AE loss (this work)':22s}" + "".join(f"{v:20.3f}" for v in row))

with open(OUT_JSON, "w") as f:
    json.dump({"signals": sigs, "auc": table}, f, indent=2)
print("\nwrote", OUT_JSON)
