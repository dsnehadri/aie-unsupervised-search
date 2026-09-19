#!/usr/bin/env python
"""Fit t(N) = L + N/T to the per-block batch sweeps (host_block_sweep.cpp CSVs)
and to the full-pipeline sweep, so every attention block gets a measured
steady-state per-event interval with launch overhead removed.

Reads $SAVE/block_sweep_{obj,cand,cross}.csv; writes $SAVE/block_intervals.json.
"""
import csv, json, os, sys
import numpy as np
from paths import DATA

SAVE = DATA
# TAG selects a measurement set: "" = the August 2026 vehicles; "_v5" = blocks3_v2
# (array blocks, v5 flags, preloaded feeders) and pl_attn_v3 (fabric blocks alone,
# t2k flags, 156.25 MHz).
TAG = os.environ.get("TAG", "")
FILES = {"Object attention": f"block_sweep_obj{TAG}.csv",
         "Candidate attention": f"block_sweep_cand{TAG}.csv",
         "Cross attention": f"block_sweep_cross{TAG}.csv",
         "PL Object attention": f"block_sweep_pl_obj{TAG}.csv",
         "PL Candidate attention": f"block_sweep_pl_cand{TAG}.csv",
         "PL Cross attention": f"block_sweep_pl_cross{TAG}.csv"}

def fit(path, nmin=8):
    n, t = [], []
    for r in csv.DictReader(open(path)):
        if r.get("min_ms") and r["min_ms"] not in ("WARMUP_FAIL", "ITER_FAIL"):
            n.append(int(r["n_events"])); t.append(float(r["min_ms"]))
    n, t = np.array(n, float), np.array(t, float)
    m = n >= nmin
    A = np.vstack([n[m], np.ones(m.sum())]).T
    (sl, ic), *_ = np.linalg.lstsq(A, t[m], rcond=None)
    pred = A @ [sl, ic]
    r2 = 1 - ((t[m] - pred) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()
    return dict(slope_us=sl * 1000, intercept_us=ic * 1000, ev_per_s=1000 / sl,
                r2=float(r2), n=n.tolist(), t_ms=t.tolist())

out = {}
print(f"{'block':22s}{'us/event':>10s}{'ev/s':>9s}{'fixed us':>10s}{'R^2':>9s}")
for lab, f in FILES.items():
    p = os.path.join(SAVE, f)
    if not os.path.isfile(p):
        print(f"{lab:22s}   (missing {f})"); continue
    d = fit(p); out[lab] = d
    print(f"{lab:22s}{d['slope_us']:10.1f}{d['ev_per_s']:9.0f}{d['intercept_us']:10.0f}{d['r2']:9.5f}")
json.dump(out, open(os.path.join(SAVE, f"block_intervals{TAG}.json"), "w"), indent=1)
print(f"wrote block_intervals{TAG}.json")
