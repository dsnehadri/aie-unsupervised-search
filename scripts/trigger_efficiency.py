#!/usr/bin/env python
"""Simple single-jet-pT trigger: object-level & event-level efficiency turn-on curves.

Trigger (one number T): a jet fires if its ONLINE pT exceeds T.
  online pT = offline pT * (1 + N(0, RES))   # RES=10% = standard trigger-resolution emulation
Object efficiency  : P(jet fires)         -> turn-on vs jet pT
Event  efficiency  : P(>=1 jet fires)     -> turn-on vs HT  ("did any jet pass")
"""
import os, numpy as np, h5py, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

T   = float(os.environ.get("T", "300"))     # trigger threshold [GeV] -- the ONE number
RES = float(os.environ.get("RES", "0.10"))  # online pT resolution (standard turn-on emulation)
rng = np.random.default_rng(0)
IN = "/home/snehadri/repos/unsupervised-search/inputs"

SAMPLES = [
    ("qcd_background",         "QCD background",                         "black"),
    ("gluino_rpv_6j",          r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",         r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",           r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000",     r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
]

def load(f):
    with h5py.File(f"{IN}/{f}.h5") as h:
        pt = np.array(h['source']['pt'])/1000.
    return np.nan_to_num(pt)                       # (N,12) GeV, zero-padded

def fires(pt):
    """online pT with resolution smear; padded (pt==0) never fires."""
    online = pt * (1.0 + RES*rng.standard_normal(pt.shape))
    return (online > T) & (pt > 0)

def binned_eff(x, passed, edges):
    """efficiency of `passed` vs x, per bin, with binomial error."""
    idx = np.digitize(x, edges) - 1
    c, p, e = [], [], []
    for b in range(len(edges)-1):
        m = idx == b
        n = m.sum()
        if n < 15:
            c.append(np.nan); p.append(np.nan); e.append(0); continue
        k = passed[m].sum(); eff = k/n
        c.append(0.5*(edges[b]+edges[b+1])); p.append(eff); e.append(np.sqrt(eff*(1-eff)/n))
    return np.array(c), np.array(p), np.array(e)

# ---------------- compute ----------------
obj_pt, obj_fire = [], []                # pooled per-jet (background) for the object turn-on
rows = []                                # single-number efficiencies per sample
evt = {}                                 # per-sample event turn-on vs HT
for f, lab, col in SAMPLES:
    pt = load(f); fr = fires(pt); real = pt > 0
    o_eff = fr.sum()/real.sum()                          # object-level single number
    ev_fire = fr.any(1)                                  # >=1 jet fires
    e_eff = ev_fire.mean()                               # event-level single number
    HT = pt.sum(1)
    rows.append((lab, col, o_eff, e_eff))
    evt[f] = (HT, ev_fire, lab, col)
    if f in ("qcd_background", "squark_rpv_8j_2000"):    # show object turn-on for 2 samples
        obj_pt.append((pt[real], fr[real], lab, col))

# ---------------- plot ----------------
plt.rcParams.update({"font.size":12})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.4))

# (a) OBJECT-LEVEL turn-on: eff vs jet pT
pedges = np.linspace(0, 900, 37)
for pt1, fr1, lab, col in obj_pt:
    c, p, e = binned_eff(pt1, fr1, pedges)
    a1.errorbar(c, p, yerr=e, fmt="o-", ms=4, lw=1.6, color=col, capsize=2, label=lab)
a1.axvline(T, color="#888", ls="--", lw=1.2); a1.text(T+10, 0.06, f"T = {T:.0f} GeV", color="#666", fontsize=11)
a1.set_xlim(0,900); a1.set_ylim(-0.02,1.05)
a1.set_xlabel("jet $p_T$  [GeV]", fontsize=13); a1.set_ylabel("object trigger efficiency", fontsize=13)
a1.xaxis.set_minor_locator(AutoMinorLocator(4)); a1.yaxis.set_minor_locator(AutoMinorLocator(5))
a1.tick_params(which="both",direction="in",right=True,top=True); a1.grid(alpha=.12)
a1.legend(frameon=False, fontsize=10.5, loc="lower right")
a1.set_title(r"(a) object level  —  $\varepsilon$(jet fires) vs $p_T$", fontsize=12.5, loc="left")

# (b) EVENT-LEVEL turn-on: eff vs HT, per sample
hedges = np.linspace(800, 4000, 33)
for f, lab, col in [(s[0],s[1],s[2]) for s in SAMPLES]:
    HT, evf, lab, col = evt[f]
    c, p, e = binned_eff(HT, evf, hedges)
    a2.errorbar(c, p, yerr=e, fmt="o-", ms=3.5, lw=1.5, color=col, capsize=1.5, label=lab)
a2.set_xlim(800,4000); a2.set_ylim(0,1.03)
a2.set_xlabel(r"$H_T = \sum p_T^{\,jet}$  [GeV]", fontsize=13); a2.set_ylabel(r"event trigger efficiency  ($\geq$1 jet fires)", fontsize=13)
a2.xaxis.set_minor_locator(AutoMinorLocator(4)); a2.yaxis.set_minor_locator(AutoMinorLocator(5))
a2.tick_params(which="both",direction="in",right=True,top=True); a2.grid(alpha=.12)
a2.legend(frameon=False, fontsize=10, loc="lower right")
a2.set_title(r"(b) event level  —  $\varepsilon$($\geq$1 jet $p_T^{online}>T$) vs $H_T$", fontsize=12.5, loc="left")

fig.suptitle(f"Single-jet-$p_T$ trigger turn-on  (T = {T:.0f} GeV, {RES*100:.0f}% online resolution)",
             fontsize=14, fontweight="bold", y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/trigger_efficiency.png"
fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"), bbox_inches="tight")
print("saved", out)
print(f"\n{'sample':34s} {'obj eff':>8s} {'event eff':>10s}   (T={T:.0f} GeV)")
for lab, col, o, e in rows:
    lab2 = lab.replace(r'$\tilde','').replace('$','').replace('\\to','->')[:32]
    print(f"{lab2:34s} {o:8.3f} {e:10.3f}")
