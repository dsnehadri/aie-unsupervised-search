#!/usr/bin/env python
"""Object-level ROC / AUC for the single-jet-pT trigger.

Score = online jet pT (offline * (1 + N(0, RES))); positive class = signal
jets, negative = QCD jets. One ROC per signal sample, AUC in the legend.
"""
import os, numpy as np, h5py, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

RES = float(os.environ.get("RES", "0.10"))
rng = np.random.default_rng(0)
IN = "/home/snehadri/repos/unsupervised-search/inputs"

SIGNALS = [
    ("gluino_rpv_6j",      r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000", r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
]

def load_online(f):
    with h5py.File(f"{IN}/{f}.h5") as h:
        pt = np.nan_to_num(np.array(h['source']['pt']) / 1000.)
    pt = pt[pt > 0]
    return pt * (1.0 + RES * rng.standard_normal(pt.shape))

def roc(sig, bkg):
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])
    order = np.argsort(-scores)
    labels = labels[order]
    tpr = np.cumsum(labels) / len(sig)
    fpr = np.cumsum(1 - labels) / len(bkg)
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc

bkg = load_online("qcd_background")

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(6.8, 6.2))
for f, lab, col in SIGNALS:
    fpr, tpr, auc = roc(load_online(f), bkg)
    ax.plot(fpr, tpr, "-", lw=1.8, color=col, label=f"{lab}  (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888")

ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("QCD jet efficiency  (false positive rate)", fontsize=13)
ax.set_ylabel("signal jet efficiency  (true positive rate)", fontsize=13)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(which="both", direction="in", right=True, top=True)
ax.grid(alpha=.12)
ax.legend(frameon=False, fontsize=10.5, loc="lower right")
ax.set_title(f"Object-level ROC: online jet $p_T$ as discriminant\n"
             f"(signal vs QCD jets, {RES*100:.0f}% online resolution)",
             fontsize=12.5)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/trigger_object_auc.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
