#!/usr/bin/env python
"""Purity / efficiency / F1 vs autoencoder-loss threshold, two rows:
top = event-level discriminant (sum of the two candidate losses),
bottom = maximum-candidate discriminant (event fires if either candidate
exceeds the threshold). Same five signals, equal event yields, same style
as the separate figures (ae_loss_metrics.py / ae_loss_percand_metrics.py).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

SAVE = "/home/snehadri/aie_scratch_save_20260810"
ev = np.load(f"{SAVE}/ae_losses.npz")
cand = np.load(f"{SAVE}/ae_losses_cand.npz")

SIGNALS = [
    ("gluino_rpv_6j",      r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000", r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
    ("squark_rpv_8j_WZH_2000", r"$\tilde q\tilde q\to 2\times jj(jj)$",  "#d62728"),
]

def curves(data, flat):
    bkg = flat(data["qcd_background"])
    allv = np.concatenate([bkg] + [flat(data[f]) for f, _, _ in SIGNALS])
    T = np.quantile(allv, np.linspace(0.0, 0.999, 200))
    out = []
    for f, lab, col in SIGNALS:
        sig = flat(data[f])
        eff = np.array([(sig > t).mean() for t in T])
        bfr = np.array([(bkg > t).mean() for t in T])
        pur = np.where(eff + bfr > 0, eff / (eff + bfr), np.nan)     # equal yields
        f1 = np.where(pur + eff > 0, 2 * pur * eff / (pur + eff), np.nan)
        out.append((lab, col, pur, eff, f1))
    return T, out

rows = [("Autoencoder loss threshold",                   curves(ev,   lambda a: a)),
        ("Maximum-candidate autoencoder loss threshold", curves(cand, lambda a: a.max(axis=1)))]
ylabels = ["Event purity", "Event efficiency", "Event $F_1$"]
letters = "abcdef"

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.4))
for r, (xlab, (T, cv)) in enumerate(rows):
    for c in range(3):
        ax = axes[r, c]
        for lab, col, pur, eff, f1 in cv:
            ax.plot(T, (pur, eff, f1)[c], "-", lw=1.8, color=col, label=lab)
        ax.set_xlim(T[0], T[-1]); ax.set_ylim(0, 1.03)
        ax.set_xlabel(xlab, fontsize=13)
        ax.set_ylabel(ylabels[c], fontsize=13)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", direction="in", right=True, top=True)
        ax.grid(alpha=.12)
        ax.text(0.02, 0.05, f"({letters[3*r+c]})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
axes[0, 0].legend(frameon=False, fontsize=10.5, loc="lower right")
fig.tight_layout(h_pad=2.0)
out = "/home/snehadri/repos/aie-unsupervised-search/figs/ae_loss_purity_f1_combined.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
