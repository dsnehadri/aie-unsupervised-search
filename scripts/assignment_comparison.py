#!/usr/bin/env python
"""Compare Passwd-ABC jet assignment to standard combinatorial algorithms.

No truth labels (unsupervised, like the paper) -> judge by the reconstructed
average parent mass: signal resolution (FWHM), background sculpting, separation.

Standard algorithms (brute force over 2-group partitions of the >=6 jets):
  - min mass asymmetry  |m1-m2|/(m1+m2)   (the paper's baseline)
  - min mass difference  |m1-m2|
  - min dR-sum           (group angularly-compact jets)
  - hemisphere           (seeded thrust-like axes, O(N) -- the standard expt method)
vs Passwd-ABC (learned, released weights) taken from figdata.npz (Fig 3 pipeline).
"""
import os, itertools, numpy as np, h5py, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

IN="/home/snehadri/repos/unsupervised-search/inputs"
SIG="gluino_rpv_6j"; SIGLAB=r"$\tilde g\tilde g\to 2\times j(jj)$ (gluino 6j, 1500 GeV)"
NBKG=12000
rng=np.random.default_rng(0)

def load(f,n=None):
    with h5py.File(f"{IN}/{f}.h5") as h:
        pt=np.nan_to_num(np.array(h['source']['pt']))/1000.
        e =np.nan_to_num(np.array(h['source']['e']))/1000.
        eta=np.array(h['source']['eta']); phi=np.array(h['source']['phi'])
    px=pt*np.cos(phi); py=pt*np.sin(phi); pz=pt*np.sinh(eta)
    p4=np.stack([e,px,py,pz],-1)                    # (N,12,4) GeV
    njet=(pt>0).sum(1); p4=p4[njet>=6]; eta=eta[njet>=6]; phi=phi[njet>=6]; pt=pt[njet>=6]
    if n and p4.shape[0]>n:
        i=rng.choice(p4.shape[0],n,replace=False); p4=p4[i]; eta=eta[i]; phi=phi[i]; pt=pt[i]
    return p4, eta, phi, pt

def m(p):
    m2=p[...,0]**2-p[...,1]**2-p[...,2]**2-p[...,3]**2
    return np.sqrt(np.clip(m2,0,None))

def brute(p4, eta, phi, pt):
    """per event: mavg for min-asym, min-diff, min-dRsum partitions."""
    N=p4.shape[0]; njet=(pt>0).sum(1)
    out={k:np.full(N,np.nan) for k in ("asym","diff","dr")}
    for n in np.unique(njet):
        n=int(n)
        if n<2: continue
        masks=[]
        for r in range(0,n):
            for extra in itertools.combinations(range(1,n),r):
                mm=np.zeros(n,bool); mm[0]=True
                for x in extra: mm[x]=True
                masks.append(mm)
        masks=np.array(masks)                        # (P,n) group-A membership
        sel=np.where(njet==n)[0]
        if len(sel)==0: continue
        jp=p4[sel][:,:n,:]; je=eta[sel][:,:n]; jf=phi[sel][:,:n]
        A=np.einsum('pn,enf->epf',masks.astype(float),jp)     # (E,P,4)
        B=jp.sum(1)[:,None,:]-A
        mA=m(A); mB=m(B)                              # (E,P)
        asym=np.abs(mA-mB)/(mA+mB+1e-9); diff=np.abs(mA-mB)
        # dR-sum: sum over jets of dR(jet, its group axis); axis = group pT-weighted mean eta/phi (approx via sum p4 direction)
        def axis_eta_phi(P):
            pxp=P[...,1]; pyp=P[...,2]; pzp=P[...,3]
            ptp=np.sqrt(pxp**2+pyp**2)+1e-9
            return np.arcsinh(pzp/ptp), np.arctan2(pyp,pxp)
        eaA,phA=axis_eta_phi(A); eaB,phB=axis_eta_phi(B)     # (E,P)
        # for each jet assign to A or B per mask, dR to that group's axis
        drsum=np.zeros(asym.shape)
        for j in range(n):
            inA=masks[:,j]                              # (P,)
            de=np.where(inA[None,:],je[:,j][:,None]-eaA, je[:,j][:,None]-eaB)
            dph=np.where(inA[None,:],jf[:,j][:,None]-phA, jf[:,j][:,None]-phB)
            dph=(dph+np.pi)%(2*np.pi)-np.pi
            drsum+=np.sqrt(de**2+dph**2)*(pt[sel][:,j][:,None]>0)
        ei=np.arange(len(sel))
        for key,metric in (("asym",asym),("diff",diff),("dr",drsum)):
            b=metric.argmin(1); out[key][sel]=0.5*(mA[ei,b]+mB[ei,b])
    return out

def hemisphere(p4, pt):
    """seeded 2-hemisphere: axis1=leading jet, axis2=jet max-mass-from-1; assign by dR, iterate."""
    N=p4.shape[0]; njet=(pt>0).sum(1); out=np.full(N,np.nan)
    def dir_ep(v):
        ptv=np.sqrt(v[...,1]**2+v[...,2]**2)+1e-9
        return np.arcsinh(v[...,3]/ptv), np.arctan2(v[...,2],v[...,1])
    for idx in range(N):
        n=int(njet[idx])
        if n<2: continue
        J=p4[idx,:n]
        # seeds: jet0 and the jet with largest invariant mass paired with jet0
        pair=m(J[0][None,:]+J)                        # mass of jet0+jetk
        s2=1+np.argmax(pair[1:]) if n>1 else 0
        a1=J[0].copy(); a2=J[s2].copy()
        for _ in range(4):
            e1,p1=dir_ep(a1); e2,p2=dir_ep(a2); ej,pj=dir_ep(J)
            d1=(ej-e1)**2+(((pj-p1+np.pi)%(2*np.pi))-np.pi)**2
            d2=(ej-e2)**2+(((pj-p2+np.pi)%(2*np.pi))-np.pi)**2
            toA=d1<=d2
            if toA.sum()==0 or toA.sum()==n: break
            a1=J[toA].sum(0); a2=J[~toA].sum(0)
        out[idx]=0.5*(m(a1)+m(a2))
    return out

def peak_fwhm(v,bins):
    v=v[np.isfinite(v)]; h,_=np.histogram(v,bins=bins); pk=h.argmax()
    peak=0.5*(bins[pk]+bins[pk+1]); half=h[pk]/2
    li=np.argmin(np.abs(h[:pk]-half)) if pk>0 else 0
    ri=np.argmin(np.abs(h[pk+1:]-half))+pk+1 if pk+1<len(h) else pk
    return peak, bins[ri]-bins[li]
def auc(b,s):
    b=b[np.isfinite(b)]; s=s[np.isfinite(s)]
    a=np.concatenate([b,s]); r=a.argsort().argsort()
    return (r[len(b):].sum()-len(s)*(len(s)+1)/2)/(len(b)*len(s))

print("loading + brute force (this takes a minute)...", flush=True)
bp4,be,bf,bpt=load("qcd_background",NBKG); sp4,se,sf,spt=load(SIG)
bb=brute(bp4,be,bf,bpt); ss=brute(sp4,se,sf,spt)
bhem=hemisphere(bp4,bpt); shem=hemisphere(sp4,spt)
# Passwd-ABC (learned) from figdata (released weights, Fig 3 pipeline)
fd=np.load("/tmp/usearch/figdata.npz")
b_abc=fd["qcd_background__mlast"]; s_abc=fd[SIG+"__mlast"]

# save per-event masses + aligned HT for the turn-on study
def fullHT(f):                       # HT aligned with figdata masses (njet>=6, file order)
    with h5py.File(f"{IN}/{f}.h5") as h:
        pt=np.nan_to_num(np.array(h['source']['pt']))/1000.
    return pt[(pt>0).sum(1)>=6].sum(1)
HTb_abc=fullHT("qcd_background"); HTs_abc=fullHT(SIG)
assert len(HTb_abc)==len(b_abc) and len(HTs_abc)==len(s_abc), (len(HTb_abc),len(b_abc),len(HTs_abc),len(s_abc))
np.savez("/tmp/assign_masses.npz",
         HTb=bpt.sum(1), HTs=spt.sum(1),         # subsample HT (heuristics)
         HTb_abc=HTb_abc, HTs_abc=HTs_abc,        # full-sample HT (Passwd-ABC)
         b_abc=b_abc, s_abc=s_abc,
         b_asym=bb["asym"], s_asym=ss["asym"], b_diff=bb["diff"], s_diff=ss["diff"],
         b_dr=bb["dr"], s_dr=ss["dr"], b_hem=bhem, s_hem=shem)

METH=[("Passwd-ABC (learned)", b_abc, s_abc, "#d62728"),
      ("min mass asymmetry",   bb["asym"], ss["asym"], "#1f77b4"),
      ("min mass difference",  bb["diff"], ss["diff"], "#2ca02c"),
      ("min $\\Delta R$-sum",  bb["dr"],   ss["dr"],   "#9467bd"),
      ("hemisphere",           bhem,       shem,       "#ff7f0e")]

bins=np.arange(0,3600,100)
rows=[]
for name,bv,sv,c in METH:
    sp,sfw=peak_fwhm(sv,bins); bp,_=peak_fwhm(bv,bins)
    rows.append((name,c,sp,sfw,bp,auc(bv,sv)))

# ---- plot ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.6,5.5))
for name,bv,sv,c in METH:
    lv=sv[np.isfinite(sv)]; a1.hist(lv,bins=bins,weights=np.ones_like(lv)/lv.size,histtype="step",color=c,lw=2.0,label=name)
# background for the winner + heuristic to show sculpting
for name,bv,sv,c,ls in [("Passwd-ABC bkg",b_abc,None,"#d62728",":"),("min-asym bkg",bb["asym"],None,"#1f77b4",":")]:
    lv=bv[np.isfinite(bv)]; a1.hist(lv,bins=bins,weights=np.ones_like(lv)/lv.size,histtype="step",color=c,lw=1.3,ls=ls,label=name)
a1.set_xlim(0,3500); a1.set_xlabel(r"reconstructed $m_\mathrm{avg}$  [GeV]",fontsize=13); a1.set_ylabel("fraction of events",fontsize=13)
a1.xaxis.set_minor_locator(AutoMinorLocator(5)); a1.yaxis.set_minor_locator(AutoMinorLocator(4))
a1.tick_params(which="both",direction="in",right=True,top=True); a1.grid(alpha=.12)
a1.legend(frameon=False,fontsize=9.5,loc="upper right")
a1.set_title("(a) mass reconstruction — signal (solid) vs background (dotted)",fontsize=12,loc="left")

# (b) summary: signal/background separation via reconstructed mass (higher=better)
order=sorted(rows,key=lambda r:r[5])          # ascending AUC
names=[r[0] for r in order]; aucs=[r[5] for r in order]; cols=[r[1] for r in order]
y=np.arange(len(names))
a2.barh(y,[a-0.8 for a in aucs],left=0.8,color=cols,alpha=.85,height=.6)
for yi,r in zip(y,order):
    win=" ★" if r[0].startswith("Passwd") else ""
    a2.text(r[5]+.003,yi,f"AUC {r[5]:.3f}   (bkg peak {r[4]:.0f} GeV){win}",va="center",fontsize=9.5,color="#333")
a2.set_yticks(y); a2.set_yticklabels(names,fontsize=11)
a2.set_xlabel("signal-vs-background separation, mass AUC  (higher = better)",fontsize=12)
a2.set_xlim(0.8,1.02); a2.grid(alpha=.12,axis="x")
a2.tick_params(which="both",direction="in",top=True)
a2.set_title("(b) discrimination — the learned assignment doesn't sculpt background",fontsize=11.5,loc="left")
fig.suptitle("Passwd-ABC vs standard combinatorial jet-assignment algorithms",fontsize=14,fontweight="bold",y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/assignment_comparison.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
print("saved",out)
print(f"\n{'method':24s} {'sig peak':>9s} {'sig FWHM':>9s} {'bkg peak':>9s} {'mass AUC':>9s}   (true mass 1500)")
for name,c,sp,sfw,bp,a in rows:
    nm=name.replace('$','').replace('\\Delta R','dR')
    print(f"{nm:24s} {sp:9.0f} {sfw:9.0f} {bp:9.0f} {a:9.3f}")
