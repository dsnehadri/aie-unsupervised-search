#!/usr/bin/env python
"""Turn-on curves for a reconstructed-mass trigger (mavg > MCUT), per jet-assignment
algorithm. Efficiency vs HT, for signal (turns on) and background (a good assignment
keeps it OFF; the sculpting heuristics turn it on too)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

MCUT=float(os.environ.get("MCUT","1000"))
d=np.load("/tmp/assign_masses.npz")
ALG=[("Passwd-ABC (learned)","abc","#d62728", True),
     ("min mass asymmetry","asym","#1f77b4", False),
     ("min mass difference","diff","#2ca02c", False),
     ("min $\\Delta R$-sum","dr","#9467bd", False),
     ("hemisphere","hem","#ff7f0e", False)]

def eff_vs(x, mass, edges):
    passed = mass > MCUT
    idx=np.digitize(x,edges)-1; c,p,e=[],[],[]
    for b in range(len(edges)-1):
        m=idx==b; n=m.sum()
        if n<25: c.append(np.nan);p.append(np.nan);e.append(0);continue
        k=passed[m].sum(); ef=k/n
        c.append(0.5*(edges[b]+edges[b+1])); p.append(ef); e.append(np.sqrt(ef*(1-ef)/n))
    return np.array(c),np.array(p),np.array(e)

edges=np.linspace(1000,4000,26)
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.6,5.5),sharey=True)
rows=[]
for name,k,col,star in ALG:
    HTs = d["HTs_abc"] if k=="abc" else d["HTs"]
    HTb = d["HTb_abc"] if k=="abc" else d["HTb"]
    ms, mb = d[f"s_{k}"], d[f"b_{k}"]
    cs,ps,es=eff_vs(HTs,ms,edges); cb,pb,eb=eff_vs(HTb,mb,edges)
    lw=2.4 if star else 1.6
    a1.errorbar(cs,ps,yerr=es,fmt="o-",ms=3.5,lw=lw,color=col,capsize=1.5,label=name)
    a2.errorbar(cb,pb,yerr=eb,fmt="o-",ms=3.5,lw=lw,color=col,capsize=1.5,label=name)
    rows.append((name,(ms>MCUT).mean(),(mb>MCUT).mean()))

for a,ttl,sub in [(a1,"(a) SIGNAL — mass trigger turns on","gluino 6j"),
                  (a2,"(b) BACKGROUND — a good assignment keeps it OFF","QCD (lower = better)")]:
    a.set_xlim(1000,4000); a.set_ylim(-0.03,1.05)
    a.set_xlabel(r"$H_T$  [GeV]",fontsize=13)
    a.xaxis.set_minor_locator(AutoMinorLocator(4)); a.yaxis.set_minor_locator(AutoMinorLocator(5))
    a.tick_params(which="both",direction="in",right=True,top=True); a.grid(alpha=.12)
    a.set_title(ttl,fontsize=12,loc="left")
    a.text(0.98,0.04,sub,transform=a.transAxes,ha="right",fontsize=10.5,color="#777")
a1.set_ylabel(r"trigger efficiency  $\varepsilon(m_\mathrm{avg}>%.0f\,$GeV$)$"%MCUT,fontsize=13)
a1.legend(frameon=False,fontsize=9.5,loc="lower right")
fig.suptitle(f"Reconstructed-mass trigger turn-on vs $H_T$  (cut $m_\\mathrm{{avg}} > {MCUT:.0f}$ GeV), per assignment algorithm",
             fontsize=13.5,fontweight="bold",y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/assignment_turnon.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
print("saved",out)
print(f"\n{'algorithm':24s} {'sig eff':>8s} {'bkg eff':>8s} {'S/sqrt(B)-ish':>14s}   (mavg>%.0f)"%MCUT)
for name,se,be in rows:
    nm=name.replace('$','').replace('\\Delta R','dR')
    print(f"{nm:24s} {se:8.3f} {be:8.3f} {se/max(be,1e-3)**0.5:14.2f}")
