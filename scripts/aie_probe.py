#!/usr/bin/env python
"""Cheap probe (no rebuild): test the Rule-5 / utilization hypothesis against
existing data — the per-tile AIE profile + the two measured batch points.
(a) batching amortization fit -> compute-bound ceiling & overhead fraction.
(b) where tile-time actually goes: GEMM (vector) vs softmax(softfloat)/layernorm/
    convert (scalar) -> the vector unit is starved."""
import csv, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---- (a) batching model from the two MEASURED points (make_aie_bottleneck_chart.py) ----
B1,T1 = 1, 551.0      # events-in-flight, throughput ev/s
B2,T2 = 100, 7549.0
t1,t2 = 1/T1, 1/T2                       # ms/event (per-event wall time)
t_over = (t1 - t2)/(1/B1 - 1/B2)         # fixed per-invocation overhead (ms)
t_comp = t1 - t_over/B1                   # per-event compute (ms)
ceil = 1/t_comp                           # compute-bound throughput ceiling
def thr(B): return 1/(t_comp + t_over/B)

# ---- (b) categorize tile time from the profile top-funcs ----
CAT = {"GEMM (vector MAC)":["gemm_tile"],
       "softmax / float-emul (scalar)":["f32_mul","softfloat"],
       "layernorm (scalar)":["layernorm_row"],
       "f32↔i32 convert":["f32_to_i32"]}
COL = {"GEMM (vector MAC)":"#2ca02c","softmax / float-emul (scalar)":"#d62728",
       "layernorm (scalar)":"#ff7f0e","f32↔i32 convert":"#9467bd","other / glue (main)":"#b0b0b0"}
def classify(top):
    frac={k:0.0 for k in CAT};
    for m in re.finditer(r"([A-Za-z0-9_<>,]+)=\d+\(([\d.]+)%\)", top):
        name,pct=m.group(1),float(m.group(2))
        for cat,keys in CAT.items():
            if any(k in name for k in keys): frac[cat]+=pct
    frac["other / glue (main)"]=max(0,100-sum(frac.values()))
    return frac

rows=list(csv.DictReader(open("/home/snehadri/repos/aie-unsupervised-search/scripts/aie_profile_per_tile.csv")))
busy=[float(r["busy_pct"]) for r in rows]
obj = np.mean([float(r["busy_pct"]) for r in rows if r["kernel"].startswith("obj")])
cand= np.mean([float(r["busy_pct"]) for r in rows if r["kernel"].startswith("cand")])
cross=np.mean([float(r["busy_pct"]) for r in rows if r["kernel"].startswith("cross")])
# aggregate visible time-share across ALL tiles (weighted by busy time)
agg={k:0.0 for k in list(CAT)+["other / glue (main)"]}; wsum=0
for r in rows:
    w=float(r["report_cyc"]); f=classify(r["top_funcs"])
    for k in agg: agg[k]+=f[k]*w
    wsum+=w
agg={k:v/wsum for k,v in agg.items()}

fig,(a1,a2)=plt.subplots(1,2,figsize=(13.6,5.3))
# (a)
BB=np.linspace(1,200,200); a1.plot(BB,[thr(b) for b in BB],"-",color="#1f77b4",lw=2.2,label="amortization model")
a1.axhline(ceil,ls="--",color="#333",lw=1.3); a1.text(120,ceil+120,f"compute-bound ceiling ≈ {ceil:.0f} ev/s",fontsize=9.5)
a1.plot([B1,B2],[T1,T2],"o",ms=8,color="#d62728",zorder=5,label="measured (B=1, B=100)")
a1.annotate(f"B=1: {100*t_over/(t_comp+t_over):.0f}% overhead",xy=(1,T1),xytext=(28,900),fontsize=9,
            arrowprops=dict(arrowstyle="->",color="#666"))
a1.annotate(f"B=100: {100*T2/ceil:.0f}% of ceiling",xy=(100,T2),xytext=(80,4200),fontsize=9,
            arrowprops=dict(arrowstyle="->",color="#666"))
a1.set_xlabel("events in flight (batch B)"); a1.set_ylabel("throughput [events/s]")
a1.set_title("(a) batching amortizes overhead — near the ceiling at B=100",fontsize=11.5,weight="bold",loc="left")
a1.legend(frameon=False,fontsize=9.5,loc="lower right"); a1.grid(alpha=.12); a1.set_ylim(0,ceil*1.15)

# (b)
cats=list(agg); vals=[agg[c] for c in cats]; cols=[COL[c] for c in cats]
y=np.arange(len(cats))[::-1]; a2.barh(y,vals,color=cols)
for yi,v in zip(y,vals): a2.text(v+0.6,yi,f"{v:.0f}%",va="center",fontsize=10,weight="bold")
a2.set_yticks(y); a2.set_yticklabels(cats,fontsize=10.5); a2.set_xlabel("share of AIE tile-time (visible top-funcs, busy-weighted)")
a2.set_title("(b) the vector unit is starved — GEMM is a sliver",fontsize=11.5,weight="bold",loc="left")
a2.set_xlim(0,max(vals)*1.25); a2.grid(alpha=.12,axis="x")
a2.text(.98,.04,f"tile busy%: obj {obj:.0f} · cross {cross:.0f} · cand {cand:.0f}\n(cand path idle → over-provisioned)",
        transform=a2.transAxes,ha="right",va="bottom",fontsize=9,color="#666")

fig.suptitle("Probe (no rebuild): why the AIE is inefficient, and where the ceiling is  —  Passwd-ABC / VCK190",
             fontsize=13,weight="bold",y=1.0)
fig.tight_layout()
out="/home/snehadri/repos/aie-unsupervised-search/figs/aie_probe.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight")
print("saved",out)
print(f"batching model: t_comp={t_comp*1000:.1f} us/ev, t_over={t_over*1000:.0f} us/invocation")
print(f"  single-event overhead fraction = {100*t_over/(t_comp+t_over):.1f}%")
print(f"  compute-bound ceiling = {ceil:.0f} ev/s ; B=100 reaches {100*T2/ceil:.0f}% of it")
print(f"tile busy%: obj={obj:.0f} cross={cross:.0f} cand={cand:.0f}")
print("tile-time composition (visible, busy-weighted):")
for c in cats: print(f"  {c:34s} {agg[c]:5.1f}%")
