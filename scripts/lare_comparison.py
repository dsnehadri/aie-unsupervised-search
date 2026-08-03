#!/usr/bin/env python
"""Recast our all-PL vs AIE-hybrid Passwd-ABC deployment in the LARE frame
(arXiv:2604.19106). LARE = PL resource needed to match the AIE performance;
the paper normalizes 1 AIE-ML tile ~ 58 DSP58. Our board is the VC1902 (400
AIE1 tiles, ~128 int8 MAC/cyc/tile ~ 29 DSP-eq) -> show the range."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# our measured points (scripts/make_pl_vs_hybrid_charts.py)
PL  = dict(name="all-PL",     thr=478,  lat=2.09,  dsp=1055, lut=204895, tiles=0,  err=0.0,  c="#d62728")
HYB = dict(name="AIE-hybrid", thr=1118, lat=0.894, dsp=1680, lut=120809, tiles=78, err=16.9, c="#1f77b4")
DSP_AVAIL, LUT_AVAIL, TILES_AVAIL = 1968, 899840, 400
TILE_DSP_ML, TILE_DSP_AIE1 = 58, 29    # paper's AIE-ML vs our VC1902 AIE1

fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.9))

# (a) performance — the measured AIE win
a=ax[0]; xs=[0,1]
a.bar(xs,[PL["thr"],HYB["thr"]],color=[PL["c"],HYB["c"]],width=.6)
for x,c in zip(xs,[PL,HYB]): a.text(x,c["thr"]+25,f'{c["thr"]}',ha="center",weight="bold")
a.set_xticks(xs); a.set_xticklabels(["all-PL","AIE-hybrid"]); a.set_ylabel("throughput [events/s]")
a.set_title("(a) performance — AIE 2.34× (measured)",fontsize=11,weight="bold",loc="left")
a.text(.5,-.16,"latency 2.09 → 0.894 ms/ev",transform=a.transAxes,ha="center",color="#1a8a2e",weight="bold",fontsize=9.5)
a.set_ylim(0,1400)

# (b) LARE resource picture: compute in DSP-equivalents (tiles->DSP), + LUT relief
a=ax[1]
pl_dsp_eq  = PL["dsp"]
hyb_dsp_eq_ml  = HYB["dsp"] + HYB["tiles"]*TILE_DSP_ML
hyb_dsp_eq_a1  = HYB["dsp"] + HYB["tiles"]*TILE_DSP_AIE1
a.bar(0, PL["dsp"], color=PL["c"], width=.55, label="PL DSP")
a.bar(1, HYB["dsp"], color=HYB["c"], width=.55)
a.bar(1, HYB["tiles"]*TILE_DSP_AIE1, bottom=HYB["dsp"], color=HYB["c"], alpha=.45, width=.55, label="AIE tiles → DSP-eq (AIE1 ×29)")
a.bar(1, HYB["tiles"]*(TILE_DSP_ML-TILE_DSP_AIE1), bottom=HYB["dsp"]+HYB["tiles"]*TILE_DSP_AIE1,
      color=HYB["c"], alpha=.2, width=.55, label="…up to AIE-ML ×58 (paper)")
a.text(0,PL["dsp"]+80,f'{PL["dsp"]}',ha="center",weight="bold",fontsize=9.5)
a.text(1,hyb_dsp_eq_ml+80,f'{hyb_dsp_eq_a1:.0f}–{hyb_dsp_eq_ml:.0f}',ha="center",weight="bold",fontsize=9.5)
a.set_xticks([0,1]); a.set_xticklabels(["all-PL","AIE-hybrid"]); a.set_ylabel("compute in DSP-equivalents")
a.set_title("(b) LARE normalization — tiles as DSP-eq",fontsize=11,weight="bold",loc="left")
a.legend(frameon=False,fontsize=8,loc="upper left"); a.set_ylim(0,6800)

# (c) efficiency indicator: throughput per DSP-equivalent (lower for AIE = tiles underused)
a=ax[2]
eff_pl  = PL["thr"]/pl_dsp_eq
eff_hyb_a1 = HYB["thr"]/hyb_dsp_eq_a1
eff_hyb_ml = HYB["thr"]/hyb_dsp_eq_ml
a.bar([0],[eff_pl],color=PL["c"],width=.55)
a.bar([1],[eff_hyb_a1],color=HYB["c"],width=.55)
a.errorbar([1],[ (eff_hyb_a1+eff_hyb_ml)/2],yerr=[[ (eff_hyb_a1-eff_hyb_ml)/2],[0]],fmt="none",ecolor="#333",capsize=4)
a.text(0,eff_pl+.01,f'{eff_pl:.2f}',ha="center",weight="bold",fontsize=9.5)
a.text(1,eff_hyb_a1+.01,f'{eff_hyb_ml:.2f}–{eff_hyb_a1:.2f}',ha="center",weight="bold",fontsize=9.5)
a.set_xticks([0,1]); a.set_xticklabels(["all-PL","AIE-hybrid"]); a.set_ylabel("throughput per DSP-eq [ev/s]")
a.set_title("(c) efficiency — AIE tiles under-used",fontsize=11,weight="bold",loc="left")
a.text(.5,-.16,"= LARE 'efficiency indicator': naive offload\nleaves AIE tiles idle (data-movement bound)",
       transform=a.transAxes,ha="center",va="top",fontsize=8.5,color="#666")
a.set_ylim(0,.55)

fig.suptitle("Passwd-ABC on VCK190 in the LARE frame  (arXiv:2604.19106)  —  our all-PL vs AIE-hybrid",
             fontsize=13,weight="bold",y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/lare_comparison.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
print("saved",out)
print(f"PL  perf/DSP-eq = {eff_pl:.3f}")
print(f"HYB perf/DSP-eq = {eff_hyb_ml:.3f} (AIE-ML×58) .. {eff_hyb_a1:.3f} (AIE1×29)")
print(f"LUT freed by AIE offload: {PL['lut']-HYB['lut']:,} ({100*(PL['lut']-HYB['lut'])/PL['lut']:.0f}%)")
print(f"DSP-eq to match AIE thr in pure PL (LARE, linear scale from PL point): {PL['dsp']*HYB['thr']/PL['thr']:.0f}")
