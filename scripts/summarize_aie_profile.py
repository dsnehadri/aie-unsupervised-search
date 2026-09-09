#!/usr/bin/env python3
"""Summarize a Work_prof_obj* aiesimulator --profile run: per-kernel us/event."""
import re, sys, glob, os
d = sys.argv[1]; nev = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = []
for f in sorted(glob.glob(os.path.join(d, "profile_funct_*.txt"))):
    txt = open(f).read()
    # the report prints the same function table twice; keep the first block only
    parts = txt.split("          Calls  Cycles tot")
    txt = parts[0] + ("          Calls  Cycles tot" + parts[1] if len(parts) > 1 else "")
    tile = re.search(r"profile_funct_(\d+_\d+)\.txt", f).group(1)
    ent = {}
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)%\s+\d+\s+\d+\s+\d+\s+(\d+)\s+([\d.]+)%\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\S+)\s", txt, re.M):
        calls, cf, pf, cfd, pfd, name = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4)), float(m.group(5)), m.group(6)
        ent.setdefault(name, [0,0,0])
        ent[name][0]+=calls; ent[name][1]+=cf; ent[name][2]=max(ent[name][2],cfd)
    kern = [k for k in ent if re.match(r"(obj|cand|cross)_(post|attn)", k)]
    if not kern: continue
    k = kern[0]
    tot_fd = ent[k][2]*ent[k][0]//max(1,ent[k][0])  # per call
    per_ev = ent[k][2]/max(1,ent[k][0])/1.25e3   # total func+desc cycles / calls -> us/event at 1.25 GHz
    soft = sum(v[1] for n,v in ent.items() if re.match(r"(f32_|float_|softfloat_|__float|__mul|__add)", n))
    ln   = sum(v[1] for n,v in ent.items() if n=="layernorm_row")
    div  = sum(v[1] for n,v in ent.items() if "div_called" in n or n.startswith("__div") or n.startswith("__udiv"))
    gem  = sum(v[1] for n,v in ent.items() if "gemm" in n)
    tot  = sum(v[1] for v in ent.values())
    rows.append((k, tile, per_ev, 100*soft/tot, 100*ln/tot, 100*div/tot, 100*gem/tot))
rows.sort(key=lambda r: -r[2])
print(f"{'kernel':<26}{'tile':>6}{'us/event':>10}{'softfloat':>11}{'layernorm':>11}{'divide':>9}{'gemm':>8}")
for k,t,p,s,l,dv,g in rows:
    print(f"{k:<26}{t:>6}{p:>10.1f}{s:>10.0f}%{l:>10.0f}%{dv:>8.0f}%{g:>7.0f}%")
print(f"slowest {rows[0][0]} = {rows[0][2]:.1f} us/event ; chain sum {sum(r[2] for r in rows):.0f} us")
