"""Tests for pure metric functions extracted from:

  - scripts/ae_loss_metrics.py          -> roc()
  - scripts/ae_loss_percand_metrics.py  -> roc()  (identical implementation)
  - scripts/auc_float_vs_int16.py       -> auc()
  - scripts/ae_ht_decorrelation_weighted.py -> wauc_setup(), wauc_from_counts(),
                                               wauc_and_err()

All four scripts execute heavy file I/O and optional model inference at import
time, so none of them is imported here.  The pure functions are reproduced
verbatim and tested against synthetic numpy arrays.

Note on auc_rank() normalization
---------------------------------
The formula  (sum_0based_signal_ranks - ns*(ns+1)/2) / (nb*ns)
uses 0-based ranks (numpy argsort().argsort()) but subtracts the 1-based
correction term.  This shifts the valid range from [0,1] to [-1/nb, 1-1/nb]:

  perfect separation → 1 - 1/nb
  perfect reversal   → -1/nb
  identical scores   → ≈ 0 (background gets lower rank on ties)

Tests use exact expected values rather than 0.0 / 1.0.
"""
import numpy as np
import pytest


# ===========================================================================
# Functions reproduced verbatim from the scripts
# ===========================================================================

# --- from ae_loss_metrics.py and ae_loss_percand_metrics.py (identical) ---

def roc(sig, bkg):
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])
    order = np.argsort(-scores)
    labels = labels[order]
    tpr = np.cumsum(labels) / len(sig)
    fpr = np.cumsum(1 - labels) / len(bkg)
    return fpr, tpr, np.trapezoid(tpr, fpr)


# --- from auc_float_vs_int16.py ---

def auc_rank(b, s):
    """auc() from auc_float_vs_int16.py (0-based rank sum formula).

    Range: [-1/nb, 1 - 1/nb].  See module docstring.
    """
    a = np.concatenate([b, s])
    r = a.argsort().argsort()
    nb, ns = len(b), len(s)
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)


# --- from ae_ht_decorrelation_weighted.py ---

def wauc_setup(bs, bwt, ss):
    o = np.argsort(bs, kind="stable")
    bs_s, bw_s = bs[o], bwt[o]
    l = np.searchsorted(bs_s, ss, "left")
    r = np.searchsorted(bs_s, ss, "right")
    return o, bw_s, l, r


def wauc_from_counts(bw_s, l, r, cb, cs):
    cw = np.concatenate([[0.0], np.cumsum(bw_s * cb)])
    W = cw[-1]
    if W <= 0:
        return np.nan
    below = 0.5 * (cw[l] + cw[r])
    return float((cs * below).sum() / (cs.sum() * W))


def wauc_and_err(bs, bwt, bsl, ss, nboot=200, seed=42):
    _rng = np.random.default_rng(seed)
    ok = np.isfinite(bs)
    bs, bwt, bsl = bs[ok], bwt[ok], bsl[ok]
    ss = ss[np.isfinite(ss)]
    nb, ns = len(bs), len(ss)
    if nb < 2 or ns < 2:
        return np.nan, np.nan, 0.0
    o, bw_s, l, r = wauc_setup(bs, bwt, ss)
    bsl_s = bsl[o]
    point = wauc_from_counts(bw_s, l, r, np.ones(nb), np.ones(ns))
    groups = [np.where(bsl_s == s)[0] for s in np.unique(bsl_s)]
    vals = np.empty(nboot)
    for b in range(nboot):
        cb = np.zeros(nb)
        for g in groups:
            cb[g] = np.bincount(_rng.integers(0, len(g), len(g)), minlength=len(g))
        cs = np.bincount(_rng.integers(0, ns, ns), minlength=ns).astype(float)
        vals[b] = wauc_from_counts(bw_s, l, r, cb, cs)
    neff = bwt.sum() ** 2 / (bwt ** 2).sum()
    return point, float(np.nanstd(vals, ddof=1)), float(neff)


# ===========================================================================
# Helpers
# ===========================================================================

def _auc_rank_expected_perfect(nb, ns):
    """For perfect separation (all sig > bkg), formula gives 1 - 1/nb."""
    return 1.0 - 1.0 / nb


def _auc_rank_expected_reversed(nb, ns):
    """For perfect reversal (all sig < bkg), formula gives -1/nb."""
    return -1.0 / nb


# ===========================================================================
# Tests: roc()
# ===========================================================================

class TestRoc:
    def test_returns_three_values(self):
        bkg = np.array([0.1, 0.3, 0.5])
        sig = np.array([0.6, 0.8, 1.0])
        result = roc(sig, bkg)
        assert len(result) == 3

    def test_auc_perfect_separation(self, perfect_sep_scores):
        bkg, sig = perfect_sep_scores
        _, _, auc = roc(sig, bkg)
        assert abs(auc - 1.0) < 1e-9, f"expected AUC=1.0, got {auc}"

    def test_auc_reversed_is_zero(self, reversed_scores):
        bkg, sig = reversed_scores
        _, _, auc = roc(sig, bkg)
        assert abs(auc - 0.0) < 1e-9, f"expected AUC≈0.0, got {auc}"

    def test_auc_in_unit_interval(self, random_scores):
        bkg, sig = random_scores
        _, _, auc = roc(sig, bkg)
        assert 0.0 <= auc <= 1.0, f"AUC out of [0,1]: {auc}"

    def test_fpr_starts_at_or_after_zero(self):
        bkg = np.array([0.1, 0.2, 0.3, 0.4])
        sig = np.array([0.5, 0.6, 0.7, 0.8])
        fpr, tpr, _ = roc(sig, bkg)
        assert fpr[0] >= 0.0

    def test_fpr_tpr_same_length(self):
        bkg = np.linspace(0, 0.5, 50)
        sig = np.linspace(0.5, 1.0, 50)
        fpr, tpr, _ = roc(sig, bkg)
        assert len(fpr) == len(tpr)
        assert len(fpr) == len(bkg) + len(sig)

    def test_larger_random_auc_better_than_half(self, rng):
        """Signal shifted higher → AUC should exceed 0.5."""
        bkg = rng.uniform(0, 1, 1000)
        sig = rng.uniform(0.3, 1.3, 1000)
        _, _, auc = roc(sig, bkg)
        assert auc > 0.5, f"expected AUC > 0.5, got {auc}"

    def test_perfect_separation_large_n(self, rng):
        """Large N: AUC should be very close to 1.0."""
        bkg = np.sort(rng.uniform(0, 1, 2000))
        sig = bkg + 2.0  # all signal strictly above all background
        _, _, auc = roc(sig, bkg)
        assert abs(auc - 1.0) < 1e-9

    def test_reversed_separation_large_n(self, rng):
        """Large N, all bkg > sig: AUC should be 0.0."""
        bkg = np.sort(rng.uniform(1, 2, 2000))
        sig = bkg - 2.0  # all signal strictly below all background
        _, _, auc = roc(sig, bkg)
        assert abs(auc - 0.0) < 1e-9


# ===========================================================================
# Tests: auc_rank()  (from auc_float_vs_int16.py)
# ===========================================================================

class TestAucRank:
    """The formula range is [-1/nb, 1-1/nb], not [0,1].

    For equal-sized groups of 5: range is [-0.2, 0.8].
    """

    def test_perfect_separation_exact(self, perfect_sep_scores):
        """All sig > bkg: result = 1 - 1/nb."""
        bkg, sig = perfect_sep_scores
        nb = len(bkg)
        expected = _auc_rank_expected_perfect(nb, len(sig))
        result = auc_rank(bkg, sig)
        assert abs(result - expected) < 1e-9, f"expected {expected}, got {result}"

    def test_reversed_exact(self, reversed_scores):
        """All bkg > sig: result = -1/nb."""
        bkg, sig = reversed_scores
        nb = len(bkg)
        expected = _auc_rank_expected_reversed(nb, len(sig))
        result = auc_rank(bkg, sig)
        assert abs(result - expected) < 1e-9, f"expected {expected}, got {result}"

    def test_perfect_beats_reversed(self, perfect_sep_scores, reversed_scores):
        """Perfect-separation score must exceed reversed score."""
        bkg_p, sig_p = perfect_sep_scores
        bkg_r, sig_r = reversed_scores
        assert auc_rank(bkg_p, sig_p) > auc_rank(bkg_r, sig_r)

    def test_result_in_valid_range(self, random_scores):
        """Result must lie in [-1/nb, 1-1/nb]."""
        bkg, sig = random_scores
        nb = len(bkg)
        result = auc_rank(bkg, sig)
        assert result >= -1.0 / nb - 1e-9
        assert result <= 1.0 - 1.0 / nb + 1e-9

    def test_better_signal_gives_higher_value(self, rng):
        """Shifting signal higher monotonically increases the rank-AUC."""
        bkg = rng.uniform(0, 1, 200)
        sig_weak = bkg + 0.5
        sig_strong = bkg + 2.0
        assert auc_rank(bkg, sig_weak) < auc_rank(bkg, sig_strong)

    def test_single_element_each_perfect(self):
        """For nb=ns=1: formula gives 1-1/1=0 for perfect sep (degenerate)."""
        result = auc_rank(np.array([0.0]), np.array([1.0]))
        expected = _auc_rank_expected_perfect(1, 1)  # = 0.0
        assert abs(result - expected) < 1e-9

    def test_large_n_approaches_one_for_perfect_sep(self, rng):
        """For large N, 1-1/nb → 1.0."""
        N = 5000
        bkg = rng.uniform(0, 1, N)
        sig = bkg + 2.0
        result = auc_rank(bkg, sig)
        assert abs(result - 1.0) < 1e-3, f"expected ~1.0, got {result}"

    def test_large_n_approaches_zero_for_reversed(self, rng):
        """For large N, -1/nb → 0.0."""
        N = 5000
        bkg = rng.uniform(1, 2, N)
        sig = bkg - 2.0
        result = auc_rank(bkg, sig)
        assert abs(result - 0.0) < 1e-3, f"expected ~0.0, got {result}"

    def test_consistent_with_roc_auc_for_large_n(self, rng):
        """For large N, rank-AUC and trapezoid AUC should agree closely."""
        N = 2000
        bkg = rng.uniform(0, 1, N)
        sig = rng.uniform(0.2, 1.2, N)
        rank_auc = auc_rank(bkg, sig)
        _, _, trap_auc = roc(sig, bkg)
        assert abs(rank_auc - trap_auc) < 0.03, (
            f"rank_auc={rank_auc:.4f} vs trap_auc={trap_auc:.4f}"
        )


# ===========================================================================
# Tests: wauc_setup() and wauc_from_counts()
# ===========================================================================

class TestWaucSetup:
    def test_output_shapes(self):
        bs = np.array([0.2, 0.4, 0.6])
        bwt = np.ones(3)
        ss = np.array([0.3, 0.5])
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        assert len(o) == 3
        assert len(bw_s) == 3
        assert len(l) == 2
        assert len(r) == 2

    def test_background_is_sorted(self):
        bs = np.array([0.9, 0.1, 0.5])
        bwt = np.array([1.0, 2.0, 3.0])
        ss = np.array([0.3])
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        sorted_bs = bs[o]
        assert np.all(np.diff(sorted_bs) >= 0)

    def test_searchsorted_boundaries(self):
        bs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        bwt = np.ones(5)
        ss = np.array([0.0, 0.5, 1.0])
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        # Signal score 0.0 → left boundary = 0 (nothing below it)
        assert l[0] == 0
        # Signal score 1.0 → right boundary = 5 (all background below it)
        assert r[2] == 5

    def test_weights_reordered_consistently(self):
        """Weights in bw_s must follow the same permutation as sorted bs."""
        bs = np.array([0.5, 0.1, 0.9])
        bwt = np.array([10.0, 20.0, 30.0])
        ss = np.array([0.3])
        o, bw_s, _, _ = wauc_setup(bs, bwt, ss)
        expected_bw_s = bwt[o]
        np.testing.assert_array_equal(bw_s, expected_bw_s)


class TestWaucFromCounts:
    def test_perfect_separation_returns_one(self):
        """All signal above all background: weighted AUC should be 1.0."""
        bs = np.array([0.1, 0.2, 0.3])
        bwt = np.ones(3)
        ss = np.array([0.5, 0.6, 0.7])
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        result = wauc_from_counts(bw_s, l, r, np.ones(3), np.ones(3))
        assert abs(result - 1.0) < 1e-9, f"expected 1.0, got {result}"

    def test_reversed_returns_zero(self):
        """All signal below all background: weighted AUC should be 0.0."""
        bs = np.array([0.5, 0.6, 0.7])
        bwt = np.ones(3)
        ss = np.array([0.1, 0.2, 0.3])
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        result = wauc_from_counts(bw_s, l, r, np.ones(3), np.ones(3))
        assert abs(result - 0.0) < 1e-9, f"expected 0.0, got {result}"

    def test_zero_weight_returns_nan(self):
        bs = np.array([0.1, 0.2, 0.3])
        bwt = np.ones(3)
        ss = np.array([0.5])
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        result = wauc_from_counts(bw_s, l, r, np.zeros(3), np.ones(1))
        assert np.isnan(result)

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(7)
        bs = np.sort(rng.uniform(0, 1, 50))
        bwt = np.ones(50)
        ss = rng.uniform(0.2, 1.2, 30)
        o, bw_s, l, r = wauc_setup(bs, bwt, ss)
        result = wauc_from_counts(bw_s, l, r, np.ones(50), np.ones(30))
        assert 0.0 <= result <= 1.0

    def test_non_uniform_weights_affect_result(self):
        """Heavier-weighted background below signal should raise AUC closer to 1."""
        bs = np.array([0.1, 0.5])
        ss = np.array([0.3])  # signal between the two bkg points
        # Equal weights
        bwt_eq = np.ones(2)
        o_eq, bw_eq, l_eq, r_eq = wauc_setup(bs, bwt_eq, ss)
        a_eq = wauc_from_counts(bw_eq, l_eq, r_eq, np.ones(2), np.ones(1))
        # Heavy weight on the low bkg point
        bwt_lo = np.array([10.0, 1.0])
        o_lo, bw_lo, l_lo, r_lo = wauc_setup(bs, bwt_lo, ss)
        a_lo = wauc_from_counts(bw_lo, l_lo, r_lo, np.ones(2), np.ones(1))
        # Heavier lower bkg → more bkg weight below signal → higher AUC
        assert a_lo > a_eq


class TestWaucAndErr:
    def test_returns_three_values(self):
        bs = np.linspace(0.0, 0.5, 20)
        bwt = np.ones(20)
        bsl = np.zeros(20, int)
        ss = np.linspace(0.6, 1.0, 10)
        point, err, neff = wauc_and_err(bs, bwt, bsl, ss, nboot=50)
        assert np.isfinite(point)
        assert np.isfinite(err)
        assert neff > 0

    def test_perfect_separation_point_is_one(self):
        bs = np.linspace(0.0, 0.4, 30)
        bwt = np.ones(30)
        bsl = np.zeros(30, int)
        ss = np.linspace(0.6, 1.0, 20)
        point, err, _ = wauc_and_err(bs, bwt, bsl, ss, nboot=50)
        assert abs(point - 1.0) < 1e-9

    def test_nan_inputs_filtered(self):
        bs = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
        bwt = np.ones(5)
        bsl = np.zeros(5, int)
        ss = np.array([0.6, 0.7, np.nan, 0.9])
        point, err, neff = wauc_and_err(bs, bwt, bsl, ss, nboot=20)
        assert isinstance(point, float)

    def test_too_few_points_returns_nan(self):
        bs = np.array([0.5])
        bwt = np.array([1.0])
        bsl = np.array([0])
        ss = np.array([0.6])
        point, err, neff = wauc_and_err(bs, bwt, bsl, ss, nboot=10)
        assert np.isnan(point)
        assert np.isnan(err)

    def test_neff_formula_uniform_weights(self):
        """For uniform weights: neff = N."""
        N = 40
        bs = np.linspace(0, 0.4, N)
        bwt = np.ones(N)
        bsl = np.zeros(N, int)
        ss = np.linspace(0.6, 1.0, 20)
        _, _, neff = wauc_and_err(bs, bwt, bsl, ss, nboot=20)
        assert abs(neff - float(N)) < 1e-6

    def test_multislice_runs(self):
        """Ensure stratified bootstrap works with multiple slice labels."""
        rng = np.random.default_rng(99)
        bs = rng.uniform(0, 0.6, 60)
        bwt = np.ones(60)
        bsl = np.array([0] * 30 + [1] * 30)
        ss = rng.uniform(0.4, 1.0, 40)
        point, err, neff = wauc_and_err(bs, bwt, bsl, ss, nboot=50)
        assert 0.0 <= point <= 1.0
        assert err >= 0.0

    def test_bootstrap_error_positive(self):
        """Bootstrap std should be > 0 for non-degenerate data."""
        rng = np.random.default_rng(77)
        bs = rng.uniform(0, 1, 50)
        bwt = np.ones(50)
        bsl = np.zeros(50, int)
        ss = rng.uniform(0.2, 1.2, 30)
        _, err, _ = wauc_and_err(bs, bwt, bsl, ss, nboot=100)
        assert err >= 0.0
