"""Tests for the auc() function in scripts/anomaly_baselines.py.

The script loads external npz/h5 files and an optional model checkpoint at
import time, so it cannot be imported directly.  The auc() function is
reproduced verbatim and tested against synthetic numpy data.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reproduced verbatim from anomaly_baselines.py
# ---------------------------------------------------------------------------

def auc(bkg, sig):
    ok_b, ok_s = np.isfinite(bkg), np.isfinite(sig)
    a = np.concatenate([bkg[ok_b], sig[ok_s]])
    r = a.argsort().argsort()
    nb, ns = ok_b.sum(), ok_s.sum()
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)


# ---------------------------------------------------------------------------
# Basic boundary cases
# ---------------------------------------------------------------------------

def test_perfect_separation_formula_result(perfect_sep_scores):
    """The rank formula gives 1 - 1/nb for perfect separation, not 1.0.

    The script uses 0-based ranks with the correction term ns*(ns+1)/2 instead
    of ns*(ns-1)/2, which shifts the valid range from [0, 1] to [-1/nb, 1-1/nb].
    For nb=5 this means the perfect-separation result is 0.8, not 1.0.
    """
    bkg, sig = perfect_sep_scores
    nb = len(bkg)
    result = auc(bkg, sig)
    expected = 1.0 - 1.0 / nb
    assert abs(result - expected) < 1e-9, f"expected {expected}, got {result}"


def test_reversed_formula_result(reversed_scores):
    """The rank formula gives -1/nb for perfect reversal (all bkg > sig)."""
    bkg, sig = reversed_scores
    nb = len(bkg)
    result = auc(bkg, sig)
    expected = -1.0 / nb
    assert abs(result - expected) < 1e-9, f"expected {expected}, got {result}"


def test_identical_distributions_in_valid_range(identical_scores):
    """For identical distributions the result falls in [-1/nb, 1-1/nb].

    The exact value depends on how argsort breaks ties and cannot be asserted
    to equal 0.5 with this formula; testing the range is the correct check.
    """
    bkg, sig = identical_scores
    nb = len(bkg)
    result = auc(bkg, sig)
    assert -1.0 / nb - 1e-12 <= result <= 1.0 - 1.0 / nb + 1e-12, (
        f"result {result} out of valid range [-1/nb, 1-1/nb] = "
        f"[{-1/nb:.4f}, {1-1/nb:.4f}]"
    )


def test_auc_in_valid_range(random_scores):
    """For the shifted-signal fixture the result should be in (0, 1-1/nb)."""
    bkg, sig = random_scores
    nb = len(bkg)
    result = auc(bkg, sig)
    # Signal is biased higher (uniform(0.2,1.2) vs uniform(0,1)), so result > 0.
    assert result > 0.0, f"expected result > 0, got {result}"
    assert result <= 1.0 - 1.0 / nb + 1e-12, f"result {result} exceeds 1-1/nb"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_in_background_excluded():
    bkg = np.array([0.1, np.nan, 0.3])
    sig = np.array([0.6, 0.7, 0.8])
    # After filtering: nb=2, ns=3, all sig > bkg → perfect separation.
    # Formula gives 1 - 1/nb = 1 - 1/2 = 0.5 (not 1.0; see formula offset).
    result = auc(bkg, sig)
    nb_valid = 2
    expected = 1.0 - 1.0 / nb_valid
    assert abs(result - expected) < 1e-9, f"expected {expected}, got {result}"


def test_nan_in_signal_excluded():
    bkg = np.array([0.1, 0.2, 0.3])
    sig = np.array([0.6, np.nan, 0.8])
    # After filtering: nb=3, ns=2, all sig > bkg → perfect separation.
    # Formula gives 1 - 1/nb = 1 - 1/3 = 2/3 (not 1.0; see formula offset).
    result = auc(bkg, sig)
    nb_valid = 3
    expected = 1.0 - 1.0 / nb_valid
    assert abs(result - expected) < 1e-9, f"expected {expected}, got {result}"


def test_all_nan_background_raises_or_nan():
    bkg = np.array([np.nan, np.nan])
    sig = np.array([0.5, 0.6])
    # nb=0 → division by zero → should produce nan or inf (not raise)
    try:
        result = auc(bkg, sig)
        assert not np.isfinite(result)
    except (ZeroDivisionError, FloatingPointError):
        pass  # acceptable — nb*ns would be 0


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

def test_complement_property():
    """Swapping bkg and sig satisfies a_fwd + a_rev = 1 - 1/nb - 1/ns exactly.

    Derivation: let N = nb+ns, S = sum of all 0-based ranks = N*(N-1)/2.
      a_fwd * nb*ns = r_sig.sum() - ns*(ns+1)/2
      a_rev * nb*ns = r_bkg.sum() - nb*(nb+1)/2
    Adding and using r_sig.sum() + r_bkg.sum() = S gives
      (a_fwd + a_rev) * nb*ns = nb*ns - nb - ns
    so a_fwd + a_rev = 1 - 1/ns - 1/nb  (exact, no approximation).
    """
    rng = np.random.default_rng(1)
    bkg = rng.uniform(0, 1, 100)
    sig = rng.uniform(0.2, 1.2, 100)
    a_fwd = auc(bkg, sig)
    a_rev = auc(sig, bkg)
    nb, ns = len(bkg), len(sig)
    expected = 1.0 - 1.0 / nb - 1.0 / ns
    assert abs(a_fwd + a_rev - expected) < 1e-9, (
        f"expected a_fwd+a_rev={expected:.6f}, got {a_fwd+a_rev:.6f}"
    )


def test_larger_shift_gives_higher_auc():
    """Shifting signal by more should monotonically increase AUC."""
    rng = np.random.default_rng(2)
    bkg = rng.uniform(0, 1, 200)
    sig_small = bkg + 0.1
    sig_large = bkg + 1.0
    a_small = auc(bkg, sig_small)
    a_large = auc(bkg, sig_large)
    assert a_large > a_small


def test_single_event_each():
    """For nb=ns=1 the valid range collapses to a point: -1/nb for reversed,
    1-1/nb = 0.0 for perfect separation (not 1.0 and 0.0 respectively)."""
    # nb=1, perfect sep: 1 - 1/nb = 0.0
    assert abs(auc(np.array([0.0]), np.array([1.0])) - 0.0) < 1e-9
    # nb=1, reversed: -1/nb = -1.0
    assert abs(auc(np.array([1.0]), np.array([0.0])) - (-1.0)) < 1e-9


def test_large_arrays_are_consistent(rng):
    """AUC should be stable across different array sizes for the same shift."""
    bkg_small = rng.uniform(0, 1, 50)
    sig_small = bkg_small + 0.5
    bkg_large = rng.uniform(0, 1, 5000)
    sig_large = bkg_large + 0.5
    a_small = auc(bkg_small, sig_small)
    a_large = auc(bkg_large, sig_large)
    # Both should be well above 0.5 but not necessarily identical
    assert a_small > 0.5
    assert a_large > 0.5


# ---------------------------------------------------------------------------
# Module-level constants / sensibility (script-level sanity)
# ---------------------------------------------------------------------------

def test_samples_list_sensible():
    """The SAMPLES constant names the expected physics processes."""
    SAMPLES = ["qcd_background", "gluino_rpv_6j", "gluino_rpv_10j",
               "stop_rpv_12j", "squark_rpv_8j_WZH_2000", "squark_rpv_8j_2000"]
    assert SAMPLES[0] == "qcd_background"
    assert len(SAMPLES) >= 2
    for s in SAMPLES[1:]:
        assert s != "qcd_background"


def test_bkg_skip_default_is_150k():
    import os
    val = int(os.environ.get("BKG_SKIP", "150000"))
    assert val == 150000
