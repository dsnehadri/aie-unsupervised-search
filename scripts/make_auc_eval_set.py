#!/usr/bin/env python
"""Pack a labelled signal+background set for the board AND compute the software
reference losses from the SAME array, so hardware and software AUC are computed
on identical events.

Provenance matters here: earlier eval_*.bin files on the board cannot be traced
back to a specific event selection, so they cannot support an AUC comparison.
This script emits input.bin, the labels, and the torch losses together.

Run: cd ~/repos/unsupervised-search && \
     ~/miniconda3/envs/unsupervised_search/bin/python \
       ~/repos/aie-unsupervised-search/scripts/make_auc_eval_set.py
"""
import json, sys
import numpy as np, h5py, torch, struct

sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
from model_blocks import Encoder

OUT = "/home/snehadri/aie_scratch_save_20260810"
CKPT = "/home/snehadri/repos/unsupervised-search/experiments/retrained_noncollapse/finalWeights.ckpt"
N_PER = 1000                       # events per class
SIGNAL = "gluino_rpv_6j"
FRAC_BITS = 9                      # data_t = ap_fixed<16,7>, matches pack_input.py

def load(fn, n):
    with h5py.File(fn, "r") as f:
        e = np.nan_to_num(np.array(f["source"]["e"])) / 1000.
        pt = np.nan_to_num(np.array(f["source"]["pt"])) / 1000.
        with np.errstate(divide="ignore"):
            le = np.log(e); le[~np.isfinite(le)] = 0
            lp = np.log(pt); lp[~np.isfinite(lp)] = 0
        phi = np.array(f["source"]["phi"]); eta = np.array(f["source"]["eta"])
        X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], -1)
        X = X[(pt > 0).sum(1) >= 6]
    return torch.tensor(X[:n], dtype=torch.float32)

def fuse_all_batchnorms(m):
    """The exported weights are BN-fused; neutralise BN so torch matches."""
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            with torch.no_grad():
                mod.running_mean.zero_(); mod.running_var.fill_(1.0)
                if mod.weight is not None: mod.weight.fill_(1.0)
                if mod.bias is not None: mod.bias.zero_()
                mod.eps = 0.0

cfg = json.load(open("config_files/replication_config.json"))
enc = Encoder(**cfg["model"]["encoder_config"])
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
enc.load_state_dict({k.replace("Encoder.", ""): v for k, v in sd.items()})
enc.eval(); fuse_all_batchnorms(enc)

Xb = load("inputs/qcd_background.h5", N_PER)
Xs = load(f"inputs/{SIGNAL}.h5", N_PER)
X = torch.cat([Xb, Xs], 0)
labels = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xs))]).astype(np.int8)
print(f"packed {len(Xb)} background + {len(Xs)} signal = {len(X)} events")

@torch.no_grad()
def losses(X):
    out = []
    for i in range(0, X.shape[0], 512):
        x = X[i:i+512]
        w = torch.stack([x[:, :, 1], x[:, :, 2], x[:, :, 3]], -1)
        mask = (x[:, :, 0] == 0).bool()
        loss, _, _, _ = enc(x, w, mask)
        out.append(loss.numpy())
    return np.concatenate(out)

sw = losses(X)

# ---- pack exactly as pack_input.py does (ap_fixed<16,7>, AP_TRN) -------------
scale = 2.0 ** FRAC_BITS
hi, lo = (2**15 - 1) / scale, -(2**15) / scale
Xn = X.numpy()
mask = (Xn[:, :, 0] == 0)
words = []
for ev in range(Xn.shape[0]):
    for i in range(12):
        for j in range(5):
            v = np.clip(Xn[ev, i, j], lo, hi)
            words.append(int(np.floor(v * scale)) & 0xFFFF)
    for i in range(12):
        words.append(1 if mask[ev, i] else 0)
open(f"{OUT}/auc_eval_input.bin", "wb").write(struct.pack(f"{len(words)}I", *words))
np.savez(f"{OUT}/auc_eval_ref.npz", labels=labels, sw_loss=sw)

def auc(b, s):
    a = np.concatenate([b, s]); r = a.argsort().argsort()
    nb, ns = len(b), len(s)
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)

print(f"software AUC on this set: {auc(sw[labels==0], sw[labels==1]):.4f}")
print(f"wrote {OUT}/auc_eval_input.bin ({len(words)} words) and auc_eval_ref.npz")
