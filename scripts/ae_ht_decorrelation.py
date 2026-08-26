#!/usr/bin/env python
"""H_T-decorrelated AE-loss performance: AUC computed WITHIN H_T bins.

Answers "does the AE loss add information beyond event energy?": inside a
narrow H_T bin, H_T itself has no discrimination left (AUC ~ 0.5 by
construction), so any remaining AE-loss AUC is structural information
orthogonal to the energy scale. Uses the cached per-event AE losses and
H_T values (same events, same order).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

SAVE = "/home/snehadri/aie_scratch_save_20260810"
ae = np.load(f"{SAVE}/ae_losses.npz")
bl = np.load(f"{SAVE}/anomaly_baselines.npz")

SIGNALS = [
    ("gluino_rpv_6j",      r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000", r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
]

def auc(bkg, sig):
    a = np.concatenate([bkg, sig])
    r = a.argsort().argsort()
    nb, ns = len(bkg), len(sig)
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)

bkg_ht, bkg_loss = bl["qcd_background_ht"], ae["qcd_background"]
MIN_N = 30
edges = np.arange(1000, 4001, 250)          # GeV
centers = 0.5 * (edges[:-1] + edges[1:])

print(f"{'H_T bin (GeV)':16s}" + "".join(f"{s[0][:14]:>16s}" for s in [(k,) for k,_,_ in SIGNALS]))
results = {}
for f, lab, col in SIGNALS:
    s_ht, s_loss = bl[f"{f}_ht"], ae[f]
    aucs, ht_aucs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bi = (bkg_ht >= lo) & (bkg_ht < hi)
        si = (s_ht >= lo) & (s_ht < hi)
        if bi.sum() >= MIN_N and si.sum() >= MIN_N:
            aucs.append(auc(bkg_loss[bi], s_loss[si]))
            ht_aucs.append(auc(bkg_ht[bi], s_ht[si]))
            ns.append((int(bi.sum()), int(si.sum())))
        else:
            aucs.append(np.nan); ht_aucs.append(np.nan); ns.append((int(bi.sum()), int(si.sum())))
    results[f] = {"auc": aucs, "ht_auc": ht_aucs, "n": ns}

for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    row = f"{lo:4.0f}-{hi:4.0f}      "
    for f, _, _ in SIGNALS:
        v = results[f]["auc"][i]
        row += f"{v:16.3f}" if np.isfinite(v) else f"{'--':>16s}"
    print(row)

# global check: AUC after a hard H_T cut (both classes above 2 TeV)
print("\nAE-loss AUC after H_T > 2 TeV cut on both classes:")
for f, lab, col in SIGNALS:
    bi = bkg_ht > 2000
    si = bl[f"{f}_ht"] > 2000
    print(f"  {f:22s} AUC={auc(bkg_loss[bi], ae[f][si]):.3f}  "
          f"(bkg n={bi.sum()}, sig n={si.sum()})")

# ---- figure ----
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7.4, 5.2))
for f, lab, col in SIGNALS:
    a = np.array(results[f]["auc"])
    ok = np.isfinite(a)
    ax.plot(centers[ok] / 1000, a[ok], "o-", lw=1.8, ms=5, color=col, label=lab)
ax.axhline(0.5, color="#888", ls="--", lw=1)
ax.text(1.05, 0.512, "no discrimination", fontsize=9, color="#666", ha="left")
ax.set_xlim(1.0, 3.0); ax.set_ylim(0.4, 1.02)
ax.set_xlabel(r"$H_T$ bin  [TeV]", fontsize=13)
ax.set_ylabel("AE-loss AUC within bin", fontsize=13)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(which="both", direction="in", right=True, top=True)
ax.grid(alpha=.12)
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_title("AE loss discrimination at fixed event energy\n"
             r"(AUC vs QCD within $H_T$ bins; bins with $\geq$30 events per class)",
             fontsize=12)
fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/ae_auc_vs_ht.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("\nsaved", out)

with open(f"{SAVE}/ae_ht_decorrelation.json", "w") as f_:
    json.dump({k: {"auc": v["auc"], "ht_auc": v["ht_auc"], "n": v["n"]}
               for k, v in results.items()}, f_, indent=2, default=float)
