#!/usr/bin/env python
"""Side-by-side mass / loss distributions for the paper (Fig. 3 rework):
horizontal panels, no titles, plain 'Fraction of Events' y-axes, no inset
commentary. Data: paper_repro/figdata.npz (retrained, paper architecture)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

d = np.load("/home/snehadri/repos/unsupervised-search/paper_repro/figdata.npz")

SIG = [
    ("gluino_rpv_6j",          r"$XX^{1500}\to 2\times j(jj)$   [6j]",   "#1f77b4"),
    ("gluino_rpv_10j",         r"$XX^{1500}\to 2\times jj(jjj)$ [10j]",  "#ff7f0e"),
    ("stop_rpv_12j",           r"$XX^{1500}\to 2\times jjj(jjj)$[12j]",  "#2ca02c"),
    ("squark_rpv_8j_WZH_2000", r"$XX^{2000}\to 2\times jj(jj)$  [8j]",   "#d62728"),
    ("squark_rpv_8j_2000",     r"$XX^{2000}\to 2\times j(jjj)$  [8j]",   "#9467bd"),
]

def paper_axes(ax):
    ax.tick_params(which="major", length=9, width=1.2, direction="in",
                   right=True, top=True, labelsize=12, pad=7)
    ax.tick_params(which="minor", length=4, width=0.9, direction="in",
                   right=True, top=True)
    for s in ax.spines.values():
        s.set_linewidth(1.2)

def frac_hist(ax, vals, bins, color, label, lw=1.6):
    vals = vals[np.isfinite(vals)]
    w = np.ones_like(vals) / vals.size
    h, _ = np.histogram(vals, bins=bins, weights=w)
    ax.hist(bins[:-1], bins, weights=h, histtype="step",
            color=color, lw=lw, label=label)

mbins = np.arange(0, 4000, 100)
lbins = np.arange(-7, 5.6, 0.4)

fig, (axm, axl) = plt.subplots(1, 2, figsize=(13.2, 4.8))

frac_hist(axm, d["qcd_background__mlast"], mbins, "black", "Background", lw=1.8)
for key, lab, col in SIG:
    frac_hist(axm, d[key + "__mlast"], mbins, col, lab)
axm.set_xlim(0, 3500); axm.set_ylim(0, None)
axm.set_xlabel(r"$m_\mathrm{avg}$  [GeV]", fontsize=15, labelpad=6, ha="right", x=1.0)
axm.set_ylabel("Fraction of Events", fontsize=14, labelpad=6, ha="right", y=1.0)
axm.xaxis.set_major_locator(MultipleLocator(500))
axm.xaxis.set_minor_locator(AutoMinorLocator(5))
axm.yaxis.set_minor_locator(AutoMinorLocator(4))
paper_axes(axm)
axm.legend(frameon=False, fontsize=10.5, loc="upper right",
           handlelength=1.3, labelspacing=0.3)

frac_hist(axl, np.log(d["qcd_background__loss"][d["qcd_background__loss"] > 0]),
          lbins, "black", "Background", lw=1.8)
for key, lab, col in SIG:
    l = d[key + "__loss"]
    frac_hist(axl, np.log(l[l > 0]), lbins, col, lab)
axl.set_xlim(-6.5, 5.0); axl.set_ylim(0, None)
axl.set_xlabel("Log(Loss)", fontsize=15, labelpad=6, ha="right", x=1.0)
axl.set_ylabel("Fraction of Events", fontsize=14, labelpad=6, ha="right", y=1.0)
axl.xaxis.set_major_locator(MultipleLocator(2))
axl.xaxis.set_minor_locator(AutoMinorLocator(4))
axl.yaxis.set_minor_locator(AutoMinorLocator(4))
paper_axes(axl)

fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/mass_loss_distributions.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
