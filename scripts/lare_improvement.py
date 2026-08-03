#!/usr/bin/env python
"""Improving the AIE's LARE efficiency (arXiv:2604.19106) using OUR measured data.
The efficiency indicator (throughput per DSP-equivalent) was low because we ran one
event at a time -> tiles idle. Cross-event pipelining (MEASURED 7549 ev/s on the same
78 tiles) flips it: the AIE goes from below PL to well above PL on the LARE axis."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# measured configs (scripts/make_pl_vs_hybrid_charts.py, make_aie_scaling_charts.py, make_aie_bottleneck_chart.py)
PL   = dict(name="all-PL",                 thr=478,  dsp=1055, lut=204895, tiles=0,  c="#d62728")
AIE1 = dict(name="all-AIE\n(1 event live)", thr=551,  dsp=314,  lut=70607,  tiles=78, c="#8fbfe0")
AIEP = dict(name="all-AIE\n+ pipelining",   thr=7549, dsp=314,  lut=70607,  tiles=78, c="#1f77b4")
CFG=[PL,AIE1,AIEP]
T29,T58 = 29,58   # DSP-eq per tile: our VC1902 AIE1 (~29)  ..  paper's AIE-ML (58)

def dsp_eq(c,t): return c["dsp"]+c["tiles"]*t
def eff(c,t): return c["thr"]/dsp_eq(c,t)

fig,(a1,a2)=plt.subplots(1,2,figsize=(13.2,5.2))
x=np.arange(3); cols=[c["c"] for c in CFG]

# (a) throughput (log) -- pipelining = 15.8x over PL, 13.7x over single-event AIE
a1.bar(x,[c["thr"] for c in CFG],color=cols,width=.6)
for xi,c in zip(x,CFG): a1.text(xi,c["thr"]*1.05,f'{c["thr"]}',ha="center",weight="bold",fontsize=10)
a1.set_yscale("log"); a1.set_ylim(300,12000)
a1.set_xticks(x); a1.set_xticklabels([c["name"] for c in CFG]); a1.set_ylabel("throughput [events/s]  (log)")
a1.set_title("(a) throughput — same 78 tiles, just fed better",fontsize=11.5,weight="bold",loc="left")
a1.annotate("",xy=(2,7549),xytext=(1,551),arrowprops=dict(arrowstyle="->",color="#1a8a2e",lw=2))
a1.text(1.5,2100,"13.7× (pipelining,\nno extra tiles)",color="#1a8a2e",weight="bold",fontsize=9,ha="center")

# (b) LARE efficiency: throughput per DSP-equivalent (higher = AIE better vs PL)
w=.36
e29=[eff(c,T29) for c in CFG]; e58=[eff(c,T58) for c in CFG]
a2.bar(x-w/2,e29,w,color=cols,label="AIE1 ×29 (our VC1902)")
a2.bar(x+w/2,e58,w,color=cols,alpha=.5,label="AIE-ML ×58 (paper)")
a2.axhline(eff(PL,T29),color="#d62728",ls="--",lw=1.3)
a2.text(2.35,eff(PL,T29)+.05,"all-PL baseline",color="#d62728",fontsize=9,ha="right")
for xi,c in zip(x,CFG):
    a2.text(xi,max(eff(c,T29),eff(c,T58))+.08,f'{eff(c,T58):.2f}–{eff(c,T29):.2f}',ha="center",fontsize=8.5,weight="bold")
a2.set_xticks(x); a2.set_xticklabels([c["name"] for c in CFG]); a2.set_ylabel("throughput per DSP-equivalent  [ev/s]")
a2.set_title("(b) LARE efficiency indicator — pipelining flips the verdict",fontsize=11.5,weight="bold",loc="left")
a2.legend(frameon=False,fontsize=8.5,loc="upper left"); a2.set_ylim(0,3.4)

fig.suptitle("Improving the AIE in the LARE frame: cross-event pipelining (measured)  —  Passwd-ABC / VCK190",
             fontsize=13,weight="bold",y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/lare_improvement.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
print("saved",out)
for c in CFG:
    print(f"{c['name'].split(chr(10))[0]:18s} thr={c['thr']:5d}  DSPeq={dsp_eq(c,T29):.0f}-{dsp_eq(c,T58):.0f}  "
          f"perf/DSPeq={eff(c,T58):.2f}-{eff(c,T29):.2f}  (PL={eff(PL,T29):.2f})")
print(f"\nall-AIE also frees PL: DSP 1055->314 (-70%), LUT 204895->70607 (-66%)")
print(f"pipelined AIE per-DSP-eq vs PL: {eff(AIEP,T29)/eff(PL,T29):.1f}x (AIE1) .. {eff(AIEP,T58)/eff(PL,T29):.1f}x (AIE-ML)")
