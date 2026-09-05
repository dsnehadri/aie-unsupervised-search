#!/usr/bin/env python
"""H_T-binned AUC (autoencoder loss vs minimum mass asymmetry) using EVERY
generated background event with its cross-section weight, instead of the
200k weighted subsample.

The subsample keeps only 743 of 107,657 high-pThat events, which are exactly
the ones populating the high-H_T bins. A weighted AUC over all events is the
same estimator (weights are constant within a pThat slice) with up to 18x
the effective statistics where it matters. Uncertainties are a stratified
bootstrap: each slice and the signal are resampled independently.

Input: bkg_weighted.npz (build_weighted_bkg_cache.py), ae_losses.npz,
anomaly_baselines.npz. Output: figs/ae_auc_vs_ht.png (replaces the
subsample version) + ae_ht_decorrelation_weighted.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

SAVE = "/home/snehadri/aie_scratch_save_20260810"
bw = np.load(f"{SAVE}/bkg_weighted.npz")
ae = np.load(f"{SAVE}/ae_losses.npz")
bl = np.load(f"{SAVE}/anomaly_baselines.npz")
B_LOSS, B_HT, B_MAVG, B_W, B_SL = bw["loss"], bw["ht"], bw["mavg"], bw["w"], bw["slice_id"]

SIGNALS = [
    ("gluino_rpv_6j",      r"$\tilde g\tilde g\to 2\times j(jj)$",   "#1f77b4"),
    ("gluino_rpv_10j",     r"$\tilde g\tilde g\to 2\times jj(jjj)$", "#ff7f0e"),
    ("stop_rpv_12j",       r"$\tilde t\tilde t\to 2\times jjj(jjj)$","#2ca02c"),
    ("squark_rpv_8j_2000", r"$\tilde q\tilde q\to 2\times j(jjj)$",  "#9467bd"),
    ("squark_rpv_8j_WZH_2000", r"$\tilde q\tilde q\to 2\times jj(jj)$",  "#d62728"),
]
RNG = np.random.default_rng(20260905)
N_BOOT = 1000
MIN_N = 30

def wauc_setup(bs, bwt, ss):
    """Pre-sort background once; return closure data for point + bootstrap."""
    o = np.argsort(bs, kind="stable")
    bs_s, bw_s = bs[o], bwt[o]
    l = np.searchsorted(bs_s, ss, "left"); r = np.searchsorted(bs_s, ss, "right")
    return o, bw_s, l, r

def wauc_from_counts(bw_s, l, r, cb, cs):
    cw = np.concatenate([[0.0], np.cumsum(bw_s * cb)])
    W = cw[-1]
    if W <= 0: return np.nan
    below = 0.5 * (cw[l] + cw[r])
    return float((cs * below).sum() / (cs.sum() * W))

def wauc_and_err(bs, bwt, bsl, ss, nboot=N_BOOT):
    """Weighted AUC and stratified-bootstrap std. bsl = slice id per bkg event."""
    ok = np.isfinite(bs); bs, bwt, bsl = bs[ok], bwt[ok], bsl[ok]
    ss = ss[np.isfinite(ss)]
    nb, ns = len(bs), len(ss)
    if nb < 2 or ns < 2: return np.nan, np.nan, 0.0
    o, bw_s, l, r = wauc_setup(bs, bwt, ss)
    bsl_s = bsl[o]
    point = wauc_from_counts(bw_s, l, r, np.ones(nb), np.ones(ns))
    groups = [np.where(bsl_s == s)[0] for s in np.unique(bsl_s)]
    vals = np.empty(nboot)
    for b in range(nboot):
        cb = np.zeros(nb)
        for g in groups:                        # resample within each slice
            cb[g] = np.bincount(RNG.integers(0, len(g), len(g)), minlength=len(g))
        cs = np.bincount(RNG.integers(0, ns, ns), minlength=ns).astype(float)
        vals[b] = wauc_from_counts(bw_s, l, r, cb, cs)
    neff = bwt.sum() ** 2 / (bwt ** 2).sum()
    return point, float(np.nanstd(vals, ddof=1)), float(neff)

edges = np.arange(1000, 4001, 250)
centers = 0.5 * (edges[:-1] + edges[1:])
results = {}
print(f"{'bin (TeV)':12s}{'n_bkg':>8s}{'n_eff':>8s}  " + "".join(f"{s[0][:12]:>14s}" for s in SIGNALS))
for f, lab, col in SIGNALS:
    s_ht, s_loss, s_mavg = bl[f"{f}_ht"], ae[f], bl[f"{f}_mavg"]
    aucs, errs, maucs, merrs, ns, neffs = [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bi = (B_HT >= lo) & (B_HT < hi); si = (s_ht >= lo) & (s_ht < hi)
        if bi.sum() >= MIN_N and si.sum() >= MIN_N:
            a, e, ne = wauc_and_err(B_LOSS[bi], B_W[bi], B_SL[bi], s_loss[si])
            m, me, _ = wauc_and_err(B_MAVG[bi], B_W[bi], B_SL[bi], s_mavg[si])
        else:
            a = e = m = me = ne = np.nan
        aucs.append(a); errs.append(e); maucs.append(m); merrs.append(me)
        ns.append((int(bi.sum()), int(si.sum()))); neffs.append(ne)
    results[f] = {"auc": aucs, "auc_err": errs, "mavg_auc": maucs, "mavg_err": merrs,
                  "n": ns, "n_eff": neffs}
for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    f0 = SIGNALS[0][0]
    row = f"{lo/1000:.2f}-{hi/1000:.2f}   {results[f0]['n'][i][0]:8d}"
    ne = results[f0]["n_eff"][i]; row += f"{ne:8.0f}  " if np.isfinite(ne) else f"{'--':>8s}  "
    for f, _, _ in SIGNALS:
        a, e = results[f]["auc"][i], results[f]["auc_err"][i]
        row += f"{a:8.3f}±{e:.3f}" if np.isfinite(a) else f"{'--':>14s}"
    print(row)

# global weighted AUC after H_T > 2 TeV on both classes
print("\nweighted AE-loss AUC after H_T > 2 TeV cut:")
for f, lab, col in SIGNALS:
    bi = B_HT > 2000; si = bl[f"{f}_ht"] > 2000
    a, e, ne = wauc_and_err(B_LOSS[bi], B_W[bi], B_SL[bi], ae[f][si], nboot=300)
    print(f"  {f:24s} AUC={a:.3f}±{e:.3f}  (bkg n={bi.sum()}, n_eff={ne:.0f}, sig n={si.sum()})")

# ---- figure (same style as the subsample version) ----
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7.4, 5.2))
for f, lab, col in SIGNALS:
    a = np.array(results[f]["auc"]); e = np.array(results[f]["auc_err"])
    m = np.array(results[f]["mavg_auc"]); me = np.array(results[f]["mavg_err"])
    ok = np.isfinite(a)
    ax.errorbar(centers[ok] / 1000, a[ok], yerr=e[ok], fmt="o-", lw=1.8, ms=5,
                color=col, label=lab, capsize=2.5, elinewidth=1.1)
    ax.errorbar(centers[ok] / 1000, m[ok], yerr=me[ok], fmt="s--", lw=1.3, ms=4,
                color=col, alpha=0.55, capsize=2, elinewidth=0.9)
from matplotlib.lines import Line2D
style_handles = [Line2D([], [], color="#444", marker="o", ls="-", label="Autoencoder loss"),
                 Line2D([], [], color="#444", marker="s", ls="--", alpha=0.55,
                        label=r"Minimum mass asymmetry, $m_\mathrm{avg}$")]
ax.set_xlim(1.0, 3.0); ax.set_ylim(0.4, 1.02)
ax.set_xlabel(r"$H_T$ [TeV]", fontsize=13); ax.set_ylabel("AUC", fontsize=13)
ax.xaxis.set_minor_locator(AutoMinorLocator(5)); ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(which="both", direction="in", right=True, top=True)
ax.grid(alpha=.12)
leg1 = ax.legend(frameon=False, fontsize=10, loc="lower right"); ax.add_artist(leg1)
ax.legend(handles=style_handles, frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
out = "/home/snehadri/repos/aie-unsupervised-search/figs/ae_auc_vs_ht.png"
fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("\nsaved", out)
json.dump(results, open(f"{SAVE}/ae_ht_decorrelation_weighted.json", "w"), indent=1, default=float)
