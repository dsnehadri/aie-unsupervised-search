#!/usr/bin/env python
"""Maximum-candidate AE-loss metrics. The event score is MSE(c0)+MSE(c1);
here an event fires if EITHER candidate exceeds the threshold, i.e. the
discriminant is max(loss_c0, loss_c1). The decision unit is the event, so
these curves are directly comparable with the summed-loss version.

Captured with forward hooks on enc.ae_in / enc.ae_out (called once per
candidate per event); loss per candidate = mean((c - c_out)^2) as in
model_blocks.get_mse. Cached in ae_losses_cand.npz. Run:
  cd ~/repos/unsupervised-search && \
  ~/miniconda3/envs/unsupervised_search/bin/python \
    ~/repos/aie-unsupervised-search/scripts/ae_loss_percand_metrics.py
"""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

CACHE = "/home/snehadri/aie_scratch_save_20260810/ae_losses_cand.npz"
CKPT = "/home/snehadri/repos/unsupervised-search/experiments/retrained_noncollapse/finalWeights.ckpt"
N_BKG = 200000

SIGNALS = [
    ("gluino_rpv_6j",      r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000", r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
    ("squark_rpv_8j_WZH_2000", r"$\tilde q\tilde q\to 2\times jj(jj)$",  "#d62728"),
]

if not os.path.isfile(CACHE):
    import sys, torch, h5py
    sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
    from model_blocks import Encoder
    cfg = json.load(open("config_files/replication_config.json"))
    enc = Encoder(**cfg["model"]["encoder_config"])
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    enc.load_state_dict({k.replace("Encoder.", ""): v for k, v in sd.items()})
    enc.eval()

    # hooks: ae_in fires once per candidate (c0 then c1) per forward
    cin, cout = [], []
    enc.ae_in.register_forward_hook(lambda m, i, o: cin.append(i[0].detach()))
    enc.ae_out.register_forward_hook(lambda m, i, o: cout.append(o.detach()))

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

    @torch.no_grad()
    def cand_losses(X):
        """(N,2): per-candidate MSE for c0 and c1."""
        out = []
        for i in range(0, X.shape[0], 4096):
            x = X[i:i+4096]
            w = torch.stack([x[:, :, 1], x[:, :, 2], x[:, :, 3]], -1)
            mask = (x[:, :, 0] == 0).bool()
            cin.clear(); cout.clear()
            enc(x, w, mask)
            l0 = torch.mean((cin[0] - cout[0]) ** 2, -1)
            l1 = torch.mean((cin[1] - cout[1]) ** 2, -1)
            out.append(torch.stack([l0, l1], -1).numpy())
        return np.concatenate(out)

    IN = "inputs"
    arrs = {"qcd_background": cand_losses(load(f"{IN}/qcd_background.h5", N_BKG))}
    for f, _, _ in SIGNALS:
        arrs[f] = cand_losses(load(f"{IN}/{f}.h5"))
        print(f"{f:22s} N={len(arrs[f])}  median cand loss {np.median(arrs[f]):.3f}")
    np.savez(CACHE, **arrs)
    print("cached", CACHE)

d = np.load(CACHE)
bkg = d["qcd_background"].max(axis=1)   # event fires if EITHER candidate is above

allv = np.concatenate([bkg] + [d[f].max(axis=1) for f, _, _ in SIGNALS])
T_SCAN = np.quantile(allv, np.linspace(0.0, 0.999, 200))

curves = []
for f, lab, col in SIGNALS:
    sig = d[f].max(axis=1)
    eff = np.array([(sig > t).mean() for t in T_SCAN])
    bfr = np.array([(bkg > t).mean() for t in T_SCAN])
    pur = np.where(eff + bfr > 0, eff / (eff + bfr), np.nan)   # equal cand yields
    f1 = np.where(pur + eff > 0, 2 * pur * eff / (pur + eff), np.nan)
    curves.append((lab, col, pur, eff, f1))

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0), sharex=True)
ylabels = ["Event purity", "Event efficiency", "Event $F_1$"]
for ax, yl, idx in zip(axes, ylabels, range(3)):
    for lab, col, pur, eff, f1 in curves:
        ax.plot(T_SCAN, (pur, eff, f1)[idx], "-", lw=1.8, color=col, label=lab)
    ax.set_xlim(T_SCAN[0], T_SCAN[-1]); ax.set_ylim(0, 1.03)
    ax.set_xlabel("Maximum-candidate autoencoder loss threshold", fontsize=13)
    ax.set_ylabel(yl, fontsize=13)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(alpha=.12)
axes[0].legend(frameon=False, fontsize=10.5, loc="lower right")
fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/ae_loss_percand_purity_f1.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)

def roc(sig, bkg):
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])
    order = np.argsort(-scores)
    labels = labels[order]
    tpr = np.cumsum(labels) / len(sig)
    fpr = np.cumsum(1 - labels) / len(bkg)
    return fpr, tpr, np.trapezoid(tpr, fpr)

fig, ax = plt.subplots(figsize=(6.8, 6.2))
for f, lab, col in SIGNALS:
    fpr, tpr, auc = roc(d[f].max(axis=1), bkg)
    ax.plot(fpr, tpr, "-", lw=1.8, color=col, label=f"{lab}  (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("QCD event efficiency  (false positive rate)", fontsize=13)
ax.set_ylabel("signal event efficiency  (true positive rate)", fontsize=13)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(which="both", direction="in", right=True, top=True)
ax.grid(alpha=.12)
ax.legend(frameon=False, fontsize=10.5, loc="lower right")
ax.set_title("Event ROC: maximum candidate AE loss\n"
             "(signal vs QCD events, retrained model)", fontsize=12.5)
fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/ae_loss_percand_auc.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
