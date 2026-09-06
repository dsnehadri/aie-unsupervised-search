#!/usr/bin/env python
"""Paired float32-vs-int16 AUC per signal from the native full-pipeline goldens
(src/pl_stream golden_main + pl_stream_top, built with and without -DFLOAT_DATAPATH,
run in software on 2000 QCD + 2000 events per signal, identical events). Answers
"how much AUC does int16 cost": <= 0.0001 on every signal."""
import re, json, numpy as np
S="/home/snehadri/aie_scratch_save_20260810/golden_float_vs_int16"
idx=json.load(open(f"{S}/index.json"))
def load(p):
    v={}
    for l in open(p):
        m=re.match(r"GOLDEN\(all-PL\) ev(\d+): MSE=([-0-9.e+nan]+)",l)
        if m: v[int(m.group(1))]=float(m.group(2))
    n=max(v)+1; a=np.full(n,np.nan)
    for k,x in v.items(): a[k]=x
    return a
fx=load(f"{S}/out_int16.txt"); fl=load(f"{S}/out_float32.txt")
def auc(b,s):
    a=np.concatenate([b,s]); r=a.argsort().argsort(); nb,ns=len(b),len(s)
    return (r[nb:].sum()-ns*(ns+1)/2)/(nb*ns)
b0,b1=idx["qcd_background"]; rng=np.random.default_rng(5)
print(f"{'signal':24s}{'float32':>9s}{'int16':>9s}{'diff':>9s}{'± (paired boot)':>17s}")
rows=[]
for s,(a0,a1) in idx.items():
    if s=="qcd_background": continue
    bf,bx=fl[b0:b1],fx[b0:b1]; sf,sx=fl[a0:a1],fx[a0:a1]
    ok_b=np.isfinite(bf)&np.isfinite(bx); ok_s=np.isfinite(sf)&np.isfinite(sx)
    bf,bx,sf,sx=bf[ok_b],bx[ok_b],sf[ok_s],sx[ok_s]
    A_f,A_x=auc(bf,sf),auc(bx,sx)
    d=[]
    for _ in range(500):
        ib=rng.integers(0,len(bf),len(bf)); i_s=rng.integers(0,len(sf),len(sf))
        d.append(auc(bx[ib],sx[i_s])-auc(bf[ib],sf[i_s]))
    d=np.array(d); rows.append((s,A_f,A_x,A_x-A_f,d.std(ddof=1)))
    print(f"{s:24s}{A_f:9.4f}{A_x:9.4f}{A_x-A_f:+9.4f}{d.std(ddof=1):17.4f}   (n_bkg={len(bf)}, n_sig={len(sf)})")
dd=np.array([r[3] for r in rows]); print(f"\nrange of (int16 - float32): {dd.min():+.4f} to {dd.max():+.4f}; mean {dd.mean():+.4f}")
