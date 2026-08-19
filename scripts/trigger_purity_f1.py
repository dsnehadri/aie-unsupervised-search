#!/usr/bin/env python
"""Object-level trigger purity / efficiency / F1 vs threshold T.

Same trigger as trigger_efficiency.py: a jet fires if its ONLINE pT
(offline * (1 + N(0, RES))) exceeds T. Per-jet, signal vs QCD, with each
sample normalized to equal event yields:
  efficiency (recall) = fired signal jets / all signal jets
  purity (precision)  = fired signal jets / all fired jets
  F1                  = 2*P*R / (P+R)
"""
import os, numpy as np, h5py, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

RES = float(os.environ.get("RES", "0.10"))
rng = np.random.default_rng(0)
IN = "/home/snehadri/repos/unsupervised-search/inputs"

BKG = ("qcd_background", "QCD background", "black")
SIGNALS = [
    ("gluino_rpv_6j",      r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000", r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
]

def load_jets(f):
    """(jet pT [GeV] flat over real jets, n_events)"""
    with h5py.File(f"{IN}/{f}.h5") as h:
        pt = np.nan_to_num(np.array(h['source']['pt']) / 1000.)
    n_ev = pt.shape[0]
    pt = pt[pt > 0]
    return pt, n_ev

def online(pt):
    return pt * (1.0 + RES * rng.standard_normal(pt.shape))

T_SCAN = np.linspace(100, 900, 81)

bkg_pt, bkg_nev = load_jets(BKG[0])
bkg_online = online(bkg_pt)
w_b = 1.0 / bkg_nev                       # equal event yields: weight = 1/N_ev
bkg_fired_w = np.array([(bkg_online > T).sum() for T in T_SCAN]) * w_b

curves = []
for f, lab, col in SIGNALS:
    sig_pt, sig_nev = load_jets(f)
    sig_online = online(sig_pt)
    w_s = 1.0 / sig_nev
    n_all_w = len(sig_pt) * w_s
    fired_w = np.array([(sig_online > T).sum() for T in T_SCAN]) * w_s
    eff = fired_w / n_all_w
    pur = np.where(fired_w + bkg_fired_w > 0, fired_w / (fired_w + bkg_fired_w), np.nan)
    f1 = np.where(pur + eff > 0, 2 * pur * eff / (pur + eff), np.nan)
    curves.append((lab, col, pur, eff, f1))

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0), sharex=True)
panels = [("(a) purity  —  fired signal jets / all fired jets", 2),
          ("(b) efficiency  —  fired signal jets / all signal jets", 3),
          ("(c) F1", 4)]
ylabels = ["object purity", "object efficiency", "object F1"]

for ax, (title, idx), yl in zip(axes, panels, ylabels):
    for lab, col, pur, eff, f1 in curves:
        y = (pur, eff, f1)[idx - 2]
        ax.plot(T_SCAN, y, "-", lw=1.8, color=col, label=lab)
    ax.set_xlim(T_SCAN[0], T_SCAN[-1]); ax.set_ylim(0, 1.03)
    ax.set_xlabel("trigger threshold $T$  [GeV]", fontsize=13)
    ax.set_ylabel(yl, fontsize=13)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(alpha=.12)
    ax.set_title(title, fontsize=12.5, loc="left")
axes[0].legend(frameon=False, fontsize=10.5, loc="upper left")

fig.suptitle(f"Single-jet-$p_T$ trigger, object level: signal vs QCD jets  "
             f"(equal event yields, {RES*100:.0f}% online resolution)",
             fontsize=14, fontweight="bold", y=1.0)
fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/trigger_purity_f1.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
for lab, col, pur, eff, f1 in curves:
    i = np.nanargmax(f1)
    print(f"{lab[:36]:38s} best F1 {f1[i]:.3f} at T={T_SCAN[i]:.0f} GeV "
          f"(purity {pur[i]:.3f}, eff {eff[i]:.3f})")
