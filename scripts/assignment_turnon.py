#!/usr/bin/env python
"""Turn-on curves for a reconstructed-mass trigger (mavg > MCUT), per jet-assignment
algorithm. Fine-binned efficiency (points+errors) with a smooth logistic turn-on fit
overlaid so each algorithm is easy to follow. Efficiency vs HT for signal & background."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

MCUT=float(os.environ.get("MCUT","1000"))
HTMAX=float(os.environ.get("HTMAX","2800"))   # cap where all algos have stats
d=np.load("/tmp/assign_masses.npz")
ALG=[("Passwd-ABC (learned)","abc","#d62728", True),
     ("min mass asymmetry","asym","#1f77b4", False),
     ("min mass difference","diff","#2ca02c", False),
     ("min $\\Delta R$-sum","dr","#9467bd", False),
     ("hemisphere","hem","#ff7f0e", False)]

def eff_vs(x, mass, edges):
    passed=mass>MCUT; idx=np.digitize(x,edges)-1; c,p,e=[],[],[]
    for b in range(len(edges)-1):
        m=idx==b; n=m.sum()
        if n<20: continue
        k=passed[m].sum(); ef=k/n
        c.append(0.5*(edges[b]+edges[b+1])); p.append(ef); e.append(np.sqrt(ef*(1-ef)/n)+1e-3)
    return np.array(c),np.array(p),np.array(e)

def logistic(x,c,A,mu,s): return c+(A-c)/(1+np.exp(-(x-mu)/s))
def smooth_fit(c,p,e):
    try:
        popt,_=curve_fit(logistic,c,p,sigma=e,absolute_sigma=False,
                         p0=[p.min(),min(1,p.max()),c[np.argmin(np.abs(p-0.5))],200],
                         bounds=([0,0,800,30],[1,1.2,3200,1200]),maxfev=20000)
        xx=np.linspace(c.min(),c.max(),200); return xx,logistic(xx,*popt)
    except Exception:
        return None,None

edges=np.linspace(1000,HTMAX,int((HTMAX-1000)/60)+1)   # ~60 GeV bins
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.8,5.6),sharey=True)
rows=[]
for name,k,col,star in ALG:
    HTs=d["HTs_abc"] if k=="abc" else d["HTs"]; HTb=d["HTb_abc"] if k=="abc" else d["HTb"]
    ms,mb=d[f"s_{k}"],d[f"b_{k}"]
    for ax,HT,mass in [(a1,HTs,ms),(a2,HTb,mb)]:
        c,p,e=eff_vs(HT,mass,edges)
        ax.errorbar(c,p,yerr=e,fmt="o",ms=3,color=col,alpha=.35,capsize=0,lw=0,elinewidth=.8)
        xx,yy=smooth_fit(c,p,e)
        if xx is not None:
            ax.plot(xx,yy,"-",color=col,lw=2.6 if star else 1.8,label=name,
                    solid_capstyle="round",zorder=5 if star else 3)
    rows.append((name,(ms>MCUT).mean(),(mb>MCUT).mean()))

for a,ttl,sub in [(a1,"(a) SIGNAL — mass trigger turns on","gluino 6j (m = 1500 GeV)"),
                  (a2,"(b) BACKGROUND — a good assignment keeps it OFF","QCD  (lower = better)")]:
    a.set_xlim(1000,HTMAX); a.set_ylim(-0.02,1.04)
    a.set_xlabel(r"$H_T$  [GeV]",fontsize=13)
    a.xaxis.set_minor_locator(AutoMinorLocator(4)); a.yaxis.set_minor_locator(AutoMinorLocator(5))
    a.tick_params(which="both",direction="in",right=True,top=True); a.grid(alpha=.12)
    a.set_title(ttl,fontsize=12.5,loc="left")
    a.text(0.985,0.045,sub,transform=a.transAxes,ha="right",fontsize=10.5,color="#777")
a1.set_ylabel(r"trigger efficiency  $\varepsilon(m_\mathrm{avg}>%.0f\,$GeV$)$"%MCUT,fontsize=13)
a1.legend(frameon=False,fontsize=10,loc="lower right",handlelength=1.6)
fig.suptitle(f"Reconstructed-mass trigger turn-on vs $H_T$  (cut $m_\\mathrm{{avg}}>{MCUT:.0f}$ GeV)  "
             f"— smooth fits, points = fine-binned data",fontsize=13,fontweight="bold",y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/assignment_turnon.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
print("saved",out)
print(f"\n{'algorithm':24s} {'sig eff':>8s} {'bkg eff':>8s}   (mavg>%.0f, integrated)"%MCUT)
for name,se,be in rows:
    print(f"{name.replace('$','').replace(chr(92)+'Delta R','dR'):24s} {se:8.3f} {be:8.3f}")
