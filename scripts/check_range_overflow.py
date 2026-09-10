#!/usr/bin/env python3
"""Do the deployed fixed-point formats overflow on SIGNAL samples?

Section 3.1 sizes the integer bits from "the largest values measured", but the
export measured them on one sample. Signal events have harder jets, so this
runs the same activations on background and on each signal sample and compares
the observed |max| against what each format can represent.

Formats deployed (attn_block_types.h): data_t ap_fixed<16,7> holds |x| < 64,
score_t ap_fixed<16,11> holds |x| < 1024.
"""
import sys, json, numpy as np, torch, h5py
sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
from model_blocks import Encoder

REPO = "/home/snehadri/repos/unsupervised-search"
CKPT = f"{REPO}/experiments/retrained_noncollapse/finalWeights.ckpt"
H5   = "/home/snehadri/sim_software/output_h5s"
N    = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SAMPLES = [("qcd_background", "qcd_background.h5"), ("gluino_rpv_6j", "gluino_rpv_6j.h5"),
           ("gluino_rpv_10j", "gluino_rpv_10j_fixed.h5"), ("squark_rpv_8j_2000", "squark_rpv_8j_2000.h5"),
           ("stop_rpv_12j", "stop_rpv_12j.h5")]

def load(fn, n):
    with h5py.File(fn, "r") as f:
        e = np.nan_to_num(np.array(f["source"]["e"])) / 1000.
        pt = np.nan_to_num(np.array(f["source"]["pt"])) / 1000.
        with np.errstate(divide="ignore"):
            le = np.log(e); le[~np.isfinite(le)] = 0
            lp = np.log(pt); lp[~np.isfinite(lp)] = 0
        phi = np.array(f["source"]["phi"]); eta = np.array(f["source"]["eta"])
        X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], -1)
        X = X[(pt > 0).sum(1) >= 6]
    return torch.tensor(X[:n], dtype=torch.float32)

def fuse(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            with torch.no_grad():
                mod.running_mean.zero_(); mod.running_var.fill_(1.0)
                if mod.weight is not None: mod.weight.fill_(1.0)
                if mod.bias is not None: mod.bias.zero_()
                mod.eps = 0.0

cfg = json.load(open(f"{REPO}/config_files/replication_config.json"))
enc = Encoder(**cfg["model"]["encoder_config"])
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
enc.load_state_dict({k.replace("Encoder.", ""): v for k, v in sd.items()})
enc.eval(); fuse(enc)

taps = {}
hooks = []
def tap(name):
    def f(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        if torch.is_tensor(t):
            a = float(t.abs().max())
            taps[name] = max(taps.get(name, 0.0), a)
    return f
for n_, m in enc.named_modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)):
        hooks.append(m.register_forward_hook(tap(n_)))

# Attention scores are computed inside nn.MultiheadAttention, so hook its INPUT
# and redo Q.K^T with the module's own packed weights. This is the quantity the
# reviewer asks about: harder signal jets could give larger scores.
score_max = {}
def score_tap(name):
    def f(m, args, kwargs):
        # the block calls attn(query=..., key=..., value=...) by keyword
        q = kwargs.get("query", args[0] if args else None)
        k_in = kwargs.get("key", args[1] if len(args) > 1 else q)
        if q is None: return
        W = m.in_proj_weight; b = m.in_proj_bias
        E = q.shape[-1]; H = m.num_heads; d = E // H
        Q = torch.nn.functional.linear(q, W[:E], b[:E] if b is not None else None)
        K = torch.nn.functional.linear(k_in, W[E:2*E], b[E:2*E] if b is not None else None)
        B, L, _ = Q.shape
        Qh = Q.view(B, L, H, d).transpose(1, 2)
        Kh = K.view(B, -1, H, d).transpose(1, 2)
        sc = (Qh @ Kh.transpose(-2, -1)) / (d ** 0.5)
        score_max[name] = max(score_max.get(name, 0.0), float(sc.abs().max()))
    return f
for n_, m in enc.named_modules():
    if isinstance(m, torch.nn.MultiheadAttention):
        hooks.append(m.register_forward_pre_hook(score_tap(n_), with_kwargs=True))

DATA_MAX, SCORE_MAX = 64.0, 1024.0
print(f"{'sample':<20}{'raw input':>11}{'max activation':>16}{'over 64?':>10}{'max attn score':>16}")
rows = {}
kinds = {}
for label, fn in SAMPLES:
    taps.clear(); score_max.clear()
    X = load(f"{H5}/{fn}", N)
    with torch.no_grad():
        w = torch.stack([X[:, :, 1], X[:, :, 2], X[:, :, 3]], -1)
        mask = (X[:, :, 0] == 0).bool()
        enc(X, w, mask)
    amax = max(taps.values()) if taps else float("nan")
    smax = max(score_max.values()) if score_max else float("nan")
    per_kind = {}
    for k, v in score_max.items():
        kind = "cand" if "cand" in k else ("cross" if "cross" in k else "obj")
        per_kind[kind] = max(per_kind.get(kind, 0.0), v)
    kinds[label] = per_kind
    worst = max(taps, key=taps.get) if taps else "-"
    rows[label] = (float(X.abs().max()), amax, worst, smax)
    print(f"{label:<20}{float(X.abs().max()):>11.2f}{amax:>16.2f}{'YES' if amax > DATA_MAX else 'no':>10}{smax:>16.2f}")
print()
b = rows["qcd_background"][1]
sb = rows["qcd_background"][3]
for label, (_, a, worst, sm) in rows.items():
    if label == "qcd_background": continue
    print(f"{label:<20} activation {a:6.2f} ({a/b:.2f}x bkg), attn score {sm:7.2f} ({sm/sb:.2f}x bkg)")
print()
print(f"{'sample':<20}{'obj scores':>12}{'cross scores':>14}{'cand scores':>13}")
for label in rows:
    k = kinds[label]
    print(f"{label:<20}{k.get('obj',0):>12.1f}{k.get('cross',0):>14.1f}{k.get('cand',0):>13.1f}")
print("formats: obj/cross score_t Q8.7 holds |x| < 256; cand Q10.5 holds |x| < 1024")
oc = max(max(k.get('obj',0), k.get('cross',0)) for k in kinds.values())
cd = max(k.get('cand',0) for k in kinds.values())
print(f"worst obj/cross score over all samples {oc:.1f} = {100*oc/256:.0f}% of its format")
print(f"worst cand score      over all samples {cd:.1f} = {100*cd/1024:.0f}% of its format")
print(f"\nheadroom on the worst sample: activations {max(r[1] for r in rows.values())/DATA_MAX*100:.0f}% of data_t,"
      f" candidate scores {cd/1024*100:.0f}% of Q10.5, obj/cross scores {oc/256*100:.0f}% of Q8.7")
print(f"\ndata_t ap_fixed<16,7> represents |x| < {DATA_MAX:.0f}; score_t ap_fixed<16,11> < {SCORE_MAX:.0f}")
for h in hooks: h.remove()
