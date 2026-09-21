"""Tests for ROC / purity / F1 computation functions.

Scripts covered:
  plot_roc_combined.py        – roc()
  plot_purity_f1_combined.py  – curves()
  plot_mass_loss_sidebyside.py – frac_hist() / paper_axes()
  plot_validation_summary.py   – median/percentile error-bar logic

The plotting calls are not tested; only the numerical computation paths are
exercised using synthetic numpy arrays.  Importing the scripts at module level
would trigger np.load() on missing hardware data, so we inline the pure
functions instead.
"""
import numpy as np
import pytest


# ===========================================================================
# plot_roc_combined.roc
# ===========================================================================

def roc(sig, bkg):
    """Inlined from plot_roc_combined.roc."""
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])
    labels = labels[np.argsort(-scores)]
    tpr = np.cumsum(labels) / len(sig)
    fpr = np.cumsum(1 - labels) / len(bkg)
    return fpr, tpr, np.trapezoid(tpr, fpr)


def test_roc_perfect_separation():
    """Signal all above background -> AUC == 1."""
    sig = np.array([10.0, 11.0, 12.0])
    bkg = np.array([0.0, 1.0, 2.0])
    fpr, tpr, auc = roc(sig, bkg)
    assert pytest.approx(auc, abs=1e-6) == 1.0


def test_roc_worst_case():
    """Background all above signal -> AUC == 0."""
    sig = np.array([0.0, 1.0, 2.0])
    bkg = np.array([10.0, 11.0, 12.0])
    _, _, auc = roc(sig, bkg)
    assert pytest.approx(auc, abs=1e-6) == 0.0


def test_roc_random_auc_near_half():
    """Identical distributions -> AUC ≈ 0.5."""
    rng = np.random.default_rng(0)
    sig = rng.uniform(0, 1, 2000)
    bkg = rng.uniform(0, 1, 2000)
    _, _, auc = roc(sig, bkg)
    assert 0.45 < auc < 0.55


def test_roc_partial_separation():
    """Signal shifted right -> AUC clearly above 0.5."""
    rng = np.random.default_rng(1)
    sig = rng.normal(1.5, 0.5, 300)
    bkg = rng.normal(0.0, 0.5, 300)
    _, _, auc = roc(sig, bkg)
    assert auc > 0.85


def test_roc_fpr_range():
    sig = np.array([2.0, 3.0, 4.0])
    bkg = np.array([0.0, 1.0, 1.5])
    fpr, tpr, _ = roc(sig, bkg)
    assert fpr.min() >= 0.0
    assert fpr.max() <= 1.0


def test_roc_tpr_range():
    sig = np.array([2.0, 3.0, 4.0])
    bkg = np.array([0.0, 1.0, 1.5])
    fpr, tpr, _ = roc(sig, bkg)
    assert tpr.min() >= 0.0
    assert tpr.max() <= 1.0


def test_roc_monotone_fpr():
    rng = np.random.default_rng(2)
    fpr, _, _ = roc(rng.normal(1, 1, 100), rng.normal(0, 1, 100))
    assert np.all(np.diff(fpr) >= 0)


def test_roc_monotone_tpr():
    rng = np.random.default_rng(3)
    _, tpr, _ = roc(rng.normal(1, 1, 100), rng.normal(0, 1, 100))
    assert np.all(np.diff(tpr) >= 0)


def test_roc_auc_in_unit_interval():
    rng = np.random.default_rng(4)
    _, _, auc = roc(rng.normal(1, 1, 200), rng.normal(0, 1, 200))
    assert 0.0 <= auc <= 1.0


def test_roc_single_sample():
    """Edge case: one signal and one background sample."""
    sig = np.array([1.0])
    bkg = np.array([0.0])
    fpr, tpr, auc = roc(sig, bkg)
    assert pytest.approx(auc, abs=1e-6) == 1.0


# ===========================================================================
# plot_purity_f1_combined.curves
# ===========================================================================

_TEST_SIGNALS = [
    ("sig_a", "Signal A", "#1f77b4"),
    ("sig_b", "Signal B", "#ff7f0e"),
]


def curves(data, flat, signals=None):
    """Inlined from plot_purity_f1_combined.curves."""
    if signals is None:
        signals = _TEST_SIGNALS
    bkg = flat(data["qcd_background"])
    allv = np.concatenate([bkg] + [flat(data[f]) for f, _, _ in signals])
    T = np.quantile(allv, np.linspace(0.0, 0.999, 200))
    out = []
    for f, lab, col in signals:
        sig = flat(data[f])
        eff = np.array([(sig > t).mean() for t in T])
        bfr = np.array([(bkg > t).mean() for t in T])
        pur = np.where(eff + bfr > 0, eff / (eff + bfr), np.nan)
        f1 = np.where(pur + eff > 0, 2 * pur * eff / (pur + eff), np.nan)
        out.append((lab, col, pur, eff, f1))
    return T, out


def _make_data(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return {
        "qcd_background": rng.normal(0.0, 1.0, 500),
        "sig_a": rng.normal(2.0, 1.0, 500),
        "sig_b": rng.normal(3.0, 1.0, 500),
    }


def test_curves_returns_200_thresholds():
    T, cv = curves(_make_data(), lambda a: a)
    assert len(T) == 200


def test_curves_signal_count():
    _, cv = curves(_make_data(), lambda a: a)
    assert len(cv) == len(_TEST_SIGNALS)


def test_curves_efficiency_in_unit_interval():
    _, cv = curves(_make_data(), lambda a: a)
    for _, _, pur, eff, _ in cv:
        assert np.all((eff >= 0) & (eff <= 1))


def test_curves_purity_in_unit_interval():
    _, cv = curves(_make_data(), lambda a: a)
    for _, _, pur, _, _ in cv:
        valid = pur[np.isfinite(pur)]
        assert np.all((valid >= 0) & (valid <= 1))


def test_curves_f1_bounded_by_one():
    _, cv = curves(_make_data(), lambda a: a)
    for _, _, _, _, f1 in cv:
        valid = f1[np.isfinite(f1)]
        assert np.all(valid <= 1.0 + 1e-9)


def test_curves_f1_is_harmonic_mean():
    """F1 = 2*P*E/(P+E); check against values computed separately."""
    _, cv = curves(_make_data(), lambda a: a)
    for _, _, pur, eff, f1 in cv:
        ok = np.isfinite(f1)
        expected = np.where(
            (pur[ok] + eff[ok]) > 0,
            2 * pur[ok] * eff[ok] / (pur[ok] + eff[ok]),
            np.nan,
        )
        np.testing.assert_allclose(f1[ok], expected, atol=1e-12)


def test_curves_2d_flatmax():
    """max(axis=1) flattening should work for per-candidate score arrays."""
    rng = np.random.default_rng(7)
    data = {
        "qcd_background": rng.normal(0.0, 1.0, (500, 2)),
        "sig_a": rng.normal(2.0, 1.0, (500, 2)),
        "sig_b": rng.normal(3.0, 1.0, (500, 2)),
    }
    T, cv = curves(data, lambda a: a.max(axis=1))
    assert len(T) == 200
    assert len(cv) == 2


def test_curves_thresholds_increasing():
    T, _ = curves(_make_data(), lambda a: a)
    assert np.all(np.diff(T) >= 0)


# ===========================================================================
# plot_mass_loss_sidebyside.frac_hist (numerical part only)
# ===========================================================================

def frac_hist_values(vals, bins):
    """Returns normalised histogram weights (no matplotlib call)."""
    vals = vals[np.isfinite(vals)]
    w = np.ones_like(vals) / vals.size
    h, _ = np.histogram(vals, bins=bins, weights=w)
    return h


def test_frac_hist_sums_to_one():
    rng = np.random.default_rng(3)
    vals = rng.normal(1500.0, 400.0, 2000)
    bins = np.arange(0, 4000, 100)
    h = frac_hist_values(vals, bins)
    assert pytest.approx(h.sum(), abs=1e-6) == 1.0


def test_frac_hist_ignores_nan():
    vals = np.array([100.0, 200.0, np.nan, 300.0, np.inf])
    bins = np.arange(0, 500, 100)
    h = frac_hist_values(vals, bins)
    assert np.isfinite(h).all()


def test_frac_hist_shape():
    vals = np.linspace(0, 3000, 200)
    bins = np.arange(0, 4000, 100)
    h = frac_hist_values(vals, bins)
    assert len(h) == len(bins) - 1


def test_frac_hist_nonnegative():
    rng = np.random.default_rng(5)
    vals = rng.normal(1000, 500, 500)
    bins = np.arange(0, 4000, 100)
    h = frac_hist_values(vals, bins)
    assert np.all(h >= 0)


def test_frac_hist_log_bins():
    """Check that the log-loss histogram (from plot_mass_loss_sidebyside) works."""
    rng = np.random.default_rng(6)
    vals = rng.normal(0.0, 2.0, 300)
    # Bins cover ±15 (7.5σ for N(0,2)) so essentially all values fall inside.
    bins = np.arange(-15, 15.4, 0.4)
    h = frac_hist_values(vals, bins)
    assert pytest.approx(h.sum(), abs=1e-6) == 1.0


# ===========================================================================
# plot_validation_summary: median / percentile error-bar logic
# ===========================================================================

def compute_errorbars(vals):
    """Inline of the error-bar computation from plot_validation_summary."""
    med = np.median(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return med, lo, hi


def test_errorbars_basic():
    vals = np.arange(1, 101, dtype=float)
    med, lo, hi = compute_errorbars(vals)
    assert lo < med < hi


def test_errorbars_constant():
    vals = np.ones(100) * 5.0
    med, lo, hi = compute_errorbars(vals)
    assert pytest.approx(med) == 5.0
    assert pytest.approx(lo) == 5.0
    assert pytest.approx(hi) == 5.0


def test_errorbars_pct_rms_scaling():
    """Replicate the PL float-error computation pattern from the script."""
    rng = np.random.default_rng(8)
    gold = rng.normal(0, 1, (100, 12, 16))
    arr = gold + rng.normal(0, 0.01, gold.shape)
    rms = np.sqrt(np.mean(gold ** 2))
    pct = 100.0 * np.abs(arr - gold).reshape(100, -1).max(1) / rms
    med, lo, hi = compute_errorbars(pct)
    assert 0.0 < lo < med < hi
    assert hi < 10.0  # should be small for 1% noise
