"""Software reference losses for the SAME 2000 events as auc_eval_input.bin,
for any config + checkpoint. make_auc_eval_set.py packs the events and writes
the two-layer reference; this reproduces its event selection exactly (first
N_PER background with >= 6 jets, then first N_PER signal) without repacking,
so a differently trained model, e.g. one ABC layer, can be scored on the set
the board sees.
usage: cd ~/repos/unsupervised-search && python make_auc_ref_model.py <config> <ckpt> <out.npz>
"""
import json, sys
import numpy as np, h5py, torch
sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
from model_blocks import Encoder
CFG, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N_PER = 1000; SIGNAL = "gluino_rpv_6j"
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
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            with torch.no_grad():
                mod.running_mean.zero_(); mod.running_var.fill_(1.0)
                if mod.weight is not None: mod.weight.fill_(1.0)
                if mod.bias is not None: mod.bias.zero_()
                mod.eps = 0.0
cfg = json.load(open(CFG))
enc = Encoder(**cfg["model"]["encoder_config"])
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
enc.load_state_dict({k.replace("Encoder.", ""): v for k, v in sd.items()})
enc.eval()
# Do NOT neutralise BatchNorm here. The export chain (export_phase3.py) FUSES
# embed.input_bn into the first embed linear, so the hardware computes the full
# model. An earlier version of this script zeroed the BN statistics in torch
# without fusing them into the linear, which is a different model with its
# input normalisation deleted: it scored 0.9688 against the hardware's 0.9825
# with rank correlation 0.89, and that gap was wrongly read as quantisation.
# With BN active the true model scores 0.9818 and correlates 0.9967 with the
# hardware. Fusing is exact, so BN-active torch == fused hardware in float.
Xb = load("inputs/qcd_background.h5", N_PER); Xs = load(f"inputs/{SIGNAL}.h5", N_PER)
X = torch.cat([Xb, Xs], 0)
labels = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xs))]).astype(np.int8)
@torch.no_grad()
def losses(X):
    out = []
    for i in range(0, X.shape[0], 512):
        x = X[i:i+512]
        w = torch.stack([x[:, :, 1], x[:, :, 2], x[:, :, 3]], -1)
        mask = (x[:, :, 0] == 0).bool()
        loss, _, _, _ = enc(x, w, mask); out.append(loss.numpy())
    return np.concatenate(out)
sw = losses(X)
np.savez(OUT, labels=labels, sw_loss=sw)
def auc(b, s):
    a = np.concatenate([b, s]); r = a.argsort().argsort(); nb, ns = len(b), len(s)
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)
print(f"{CKPT.split('/')[-1]}: software AUC on the board's 2000-event set = {auc(sw[labels==0], sw[labels==1]):.4f}  -> {OUT}")
