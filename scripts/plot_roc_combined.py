#!/usr/bin/env python
"""Event-level and per-candidate ROC curves side by side.

Discriminant is the autoencoder reconstruction loss: summed over the two
BSM candidates for the event-level panel, and taken per candidate (two
objects per event) for the candidate-level panel. Paper style: no titles,
capitalised axis labels, no parenthetical asides.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

SAVE = "/home/snehadri/aie_scratch_save_20260810"
ev = np.load(f"{SAVE}/ae_losses.npz")
cand = np.load(f"{SAVE}/ae_losses_cand.npz")

SIGNALS = [
    ("gluino_rpv_6j",      r"$XX^{1500}\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$XX^{1500}\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$XX^{1500}\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_WZH_2000", r"$XX^{2000}\to 2\times jj(jj)$",  "#d62728"),

    ("squark_rpv_8j_2000", r"$XX^{2000}\to 2\times j(jjj)$",  "#9467bd"),
]

def roc(sig, bkg):
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])
    labels = labels[np.argsort(-scores)]
    tpr = np.cumsum(labels) / len(sig)
    fpr = np.cumsum(1 - labels) / len(bkg)
    return fpr, tpr, np.trapezoid(tpr, fpr)

plt.rcParams.update({"font.size": 12})
fig, (ax_ev, ax_cd) = plt.subplots(1, 2, figsize=(12.4, 5.8))

panels = [
    (ax_ev, ev,   lambda a: a,                "Signal event efficiency",
     "QCD event efficiency"),
    (ax_cd, cand, lambda a: a.max(axis=1),    "Signal event efficiency",
     "QCD event efficiency"),
]

for ax, data, flat, ylab, xlab in panels:
    bkg = flat(data["qcd_background"])
    for f, lab, col in SIGNALS:
        fpr, tpr, auc = roc(flat(data[f]), bkg)
        ax.plot(fpr, tpr, "-", lw=1.8, color=col, label=f"{lab}, AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(alpha=.12)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    # single 0 at the corner: the y-axis label serves both axes
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/roc_event_and_candidate.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
