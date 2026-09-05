#!/usr/bin/env python
"""Score EVERY generated QCD background event, keeping its slice weight.

The delivered inputs/qcd_background.h5 is a 200k weighted draw: it spends its
budget where the cross section is, so it keeps 163k of 171k available low-pThat
events but only 743 of 107,657 high-pThat ones. The discarded high-pThat events
are exactly those populating the high-H_T bins. Here we score all of them and
carry a per-event weight instead, which leaves the estimator unbiased but
raises the effective sample size by up to ~18x in the highest bins.

Weights are constant within a pThat slice (verified), so w = sigma / N_generated.

Run: cd ~/repos/unsupervised-search && \
     ~/miniconda3/envs/unsupervised_search/bin/python \
       ~/repos/aie-unsupervised-search/scripts/build_weighted_bkg_cache.py
"""
import glob, itertools, json, os, sys
import numpy as np
import h5py
import torch

sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
from model_blocks import Encoder, x_to_p4

OUT = "/home/snehadri/aie_scratch_save_20260810/bkg_weighted.npz"
CKPT = "/home/snehadri/repos/unsupervised-search/experiments/retrained_noncollapse/finalWeights.ckpt"
B1DIR = "/home/snehadri/sim_software/qcd_background/bin1_highstat"
BASE = "/home/snehadri/sim_software/output_h5s"

# ---------- slice weights ----------
sig, ngen = [], 0
for m in glob.glob(f"{B1DIR}/job_*.meta"):
    kv = dict(t.split("=") for t in open(m).read().split())
    ngen += int(kv["nevents"])
    s = float(kv["sigma_mb"])
    if np.isfinite(s):
        sig.append(s)
W1 = float(np.mean(sig)) / ngen

def load_raw():
    """(pt, eta, phi, e, slice_id) for every generated background event."""
    chunks = []
    p = []
    for f in sorted(glob.glob(f"{B1DIR}/job_*.h5")):
        with h5py.File(f, "r") as h:
            if h["source"]["pt"].ndim < 2 or h["source"]["pt"].shape[0] == 0:
                continue
            p.append({k: h["source"][k][:] for k in ("pt", "eta", "phi", "e")})
    b1 = {k: np.concatenate([d[k] for d in p]) for k in p[0]}
    chunks.append((b1, W1, 1))
    for b in (2, 3):
        with h5py.File(f"{BASE}/qcd_bin{b}.h5", "r") as h:
            d = {k: h["source"][k][:] for k in ("pt", "eta", "phi", "e")}
            w = float(h["EventVars"]["normweight"][0])
        chunks.append((d, w, b))
    return chunks

def to_X(d):
    e = np.nan_to_num(d["e"]) / 1000.
    pt = np.nan_to_num(d["pt"]) / 1000.
    with np.errstate(divide="ignore"):
        le = np.log(e); le[~np.isfinite(le)] = 0
        lp = np.log(pt); lp[~np.isfinite(lp)] = 0
    X = np.stack([lp, d["eta"], np.cos(d["phi"]), np.sin(d["phi"]), le], -1)
    keep = (pt > 0).sum(1) >= 6
    return torch.tensor(X[keep], dtype=torch.float32)

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
        mf = np.array(combos).astype(float)
        sel = np.where(njet == n)[0]
        if len(sel) == 0: continue
        def mass(p):
            m2 = p[..., 0]**2 - p[..., 1]**2 - p[..., 2]**2 - p[..., 3]**2
            return np.sqrt(np.clip(m2, 0, None))
        CH = 2000
        for c0 in range(0, len(sel), CH):
            idx = sel[c0:c0 + CH]
            jets = jp4[idx][:, :n, :]
            A = np.einsum("pn,enf->epf", mf, jets)
            B = jets.sum(1)[:, None, :] - A
            mA, mB = mass(A), mass(B)
            valid = (mA + mB) > 0
            asym = np.where(valid, np.abs(mA - mB) / (mA + mB + 1e-9), np.inf)
            best = asym.argmin(1)
            ei = np.arange(len(idx))
            out[idx] = 0.5 * (mA[ei, best] + mB[ei, best])
    return out

cfg = json.load(open("config_files/replication_config.json"))
enc = Encoder(**cfg["model"]["encoder_config"])
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
enc.load_state_dict({k.replace("Encoder.", ""): v for k, v in sd.items()})
enc.eval()

@torch.no_grad()
def reco_loss(X):
    out = []
    for i in range(0, X.shape[0], 4096):
        x = X[i:i+4096]
        w = torch.stack([x[:, :, 1], x[:, :, 2], x[:, :, 3]], -1)
        mask = (x[:, :, 0] == 0).bool()
        loss, _, _, _ = enc(x, w, mask)
        out.append(loss.numpy())
    return np.concatenate(out)

loss_a, ht_a, mavg_a, w_a, sl_a = [], [], [], [], []
for d, w, sid in load_raw():
    X = to_X(d)
    n = X.shape[0]
    print(f"slice {sid}: {n} events, w={w:.4e}", flush=True)
    loss_a.append(reco_loss(X))
    jp4 = x_to_p4(X).numpy()
    pt = np.sqrt(jp4[:, :, 1]**2 + jp4[:, :, 2]**2)
    pt[jp4[:, :, 0] == 0] = 0
    njet = (jp4[:, :, 0] > 0).sum(1)
    ht_a.append(pt.sum(1))
    print(f"  scored; computing min-asym for {n}...", flush=True)
    mavg_a.append(min_asym_mavg(jp4, njet))
    w_a.append(np.full(n, w))
    sl_a.append(np.full(n, sid))
    print(f"  slice {sid} done", flush=True)

np.savez(OUT,
         loss=np.concatenate(loss_a), ht=np.concatenate(ht_a),
         mavg=np.concatenate(mavg_a), w=np.concatenate(w_a),
         slice_id=np.concatenate(sl_a))
print("wrote", OUT)
