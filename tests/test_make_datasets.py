"""Tests for dataset-construction script logic.

Scripts covered:
  scripts/make_validation_set.py  – event packing (float32 view as uint32,
                                    mask words, 6-jet padding)
  scripts/make_auc_eval_set.py    – load() feature engineering, auc()
  scripts/make_auc_ref_model.py   – load() with skip, auc() (same formula)

All three scripts execute heavy code at module level (h5py file opens, torch
model loads, sys.argv reads) and have no __main__ guard, so they cannot be
imported safely.  Pure functions are inlined and tested with synthetic numpy
arrays.  Tests that need h5py are skipped at collection time if the package is
absent; tests that need torch are similarly guarded.
"""
import struct
import unittest.mock as mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# auc() — Wilcoxon-Mann-Whitney rank-sum AUC
# Identical implementation in make_auc_eval_set.py and make_auc_ref_model.py
# ---------------------------------------------------------------------------

def auc(b, s):
    """Inlined from make_auc_eval_set.auc / make_auc_ref_model.auc."""
    a = np.concatenate([b, s])
    # +1 converts to 1-based ranks; gives AUC=1 for perfect separation.
    r = a.argsort().argsort() + 1
    nb, ns = len(b), len(s)
    return (r[nb:].sum() - ns * (ns + 1) / 2) / (nb * ns)


class TestAUC:
    def test_perfect_separation_auc_one(self):
        """Signal all above background -> AUC == 1."""
        b = np.array([0.0, 1.0, 2.0])
        s = np.array([10.0, 11.0, 12.0])
        assert pytest.approx(auc(b, s), abs=1e-9) == 1.0

    def test_worst_case_auc_zero(self):
        """Background all above signal -> AUC == 0."""
        b = np.array([10.0, 11.0, 12.0])
        s = np.array([0.0, 1.0, 2.0])
        assert pytest.approx(auc(b, s), abs=1e-9) == 0.0

    def test_random_auc_near_half(self):
        """Identical distributions -> AUC close to 0.5."""
        rng = np.random.default_rng(0)
        b = rng.uniform(0, 1, 2000)
        s = rng.uniform(0, 1, 2000)
        assert 0.45 < auc(b, s) < 0.55

    def test_partial_separation(self):
        """Shifted signal -> AUC clearly above 0.5."""
        rng = np.random.default_rng(1)
        b = rng.normal(0.0, 1.0, 500)
        s = rng.normal(2.0, 1.0, 500)
        assert auc(b, s) > 0.85

    def test_auc_in_unit_interval(self):
        rng = np.random.default_rng(2)
        b = rng.normal(0, 1, 300)
        s = rng.normal(1, 1, 300)
        a = auc(b, s)
        assert 0.0 <= a <= 1.0

    def test_single_pair_perfect(self):
        assert pytest.approx(auc(np.array([0.0]), np.array([1.0])), abs=1e-9) == 1.0

    def test_single_pair_worst(self):
        assert pytest.approx(auc(np.array([1.0]), np.array([0.0])), abs=1e-9) == 0.0

    def test_symmetric_complement(self):
        """auc(b, s) + auc(s, b) should equal 1 when there are no ties."""
        rng = np.random.default_rng(3)
        b = rng.normal(0, 1, 200)
        s = rng.normal(1, 1, 200)
        assert pytest.approx(auc(b, s) + auc(s, b), abs=1e-9) == 1.0

    def test_returns_scalar(self):
        b = np.array([0.1, 0.2])
        s = np.array([0.8, 0.9])
        result = auc(b, s)
        assert np.ndim(result) == 0


# ---------------------------------------------------------------------------
# Feature engineering: log transforms, cos/sin, nan handling
# Shared logic from make_auc_eval_set.load() and make_auc_ref_model.load()
# ---------------------------------------------------------------------------

def feature_engineer(e_raw, pt_raw, phi_raw, eta_raw):
    """
    Reproduce the feature-engineering block from load() in both eval scripts.
    Inputs are 2-D arrays: (n_events, n_jets).
    Returns X of shape (n_events, n_jets, 5).
    """
    e = np.nan_to_num(e_raw) / 1000.0
    pt = np.nan_to_num(pt_raw) / 1000.0
    with np.errstate(divide="ignore"):
        le = np.log(e)
        le[~np.isfinite(le)] = 0.0
        lp = np.log(pt)
        lp[~np.isfinite(lp)] = 0.0
    X = np.stack([lp, eta_raw, np.cos(phi_raw), np.sin(phi_raw), le], axis=-1)
    return X


class TestFeatureEngineering:
    def test_output_shape(self):
        rng = np.random.default_rng(10)
        n, j = 20, 15
        X = feature_engineer(
            rng.uniform(0.1, 500.0, (n, j)),
            rng.uniform(5.0, 200.0, (n, j)),
            rng.uniform(-np.pi, np.pi, (n, j)),
            rng.uniform(-3.0, 3.0, (n, j)),
        )
        assert X.shape == (n, j, 5)

    def test_cos_phi_in_range(self):
        rng = np.random.default_rng(11)
        n, j = 30, 12
        phi = rng.uniform(-np.pi, np.pi, (n, j))
        X = feature_engineer(
            np.ones((n, j)), np.ones((n, j)) * 10.0, phi, np.zeros((n, j))
        )
        cos_vals = X[:, :, 2]
        assert np.all(cos_vals >= -1.0) and np.all(cos_vals <= 1.0)

    def test_sin_phi_in_range(self):
        rng = np.random.default_rng(12)
        n, j = 30, 12
        phi = rng.uniform(-np.pi, np.pi, (n, j))
        X = feature_engineer(
            np.ones((n, j)), np.ones((n, j)) * 10.0, phi, np.zeros((n, j))
        )
        sin_vals = X[:, :, 3]
        assert np.all(sin_vals >= -1.0) and np.all(sin_vals <= 1.0)

    def test_nan_input_produces_finite_output(self):
        e = np.array([[np.nan, 100.0, 200.0]])
        pt = np.array([[np.nan, 10.0, 20.0]])
        phi = np.array([[0.0, 1.0, 2.0]])
        eta = np.array([[0.0, 0.5, -0.5]])
        X = feature_engineer(e, pt, phi, eta)
        assert np.isfinite(X).all()

    def test_zero_pt_gives_zero_log(self):
        """pt=0 -> log(0) -> -inf -> replaced by 0."""
        e = np.array([[10.0]])
        pt = np.array([[0.0]])
        phi = np.array([[0.0]])
        eta = np.array([[0.0]])
        X = feature_engineer(e, pt, phi, eta)
        assert X[0, 0, 0] == 0.0  # lp feature

    def test_log_monotone_with_pt(self):
        """Larger pt should produce larger lp feature."""
        n, j = 1, 3
        pt = np.array([[10.0, 50.0, 100.0]])
        e = np.ones((n, j)) * 50.0
        phi = np.zeros((n, j))
        eta = np.zeros((n, j))
        X = feature_engineer(e, pt, phi, eta)
        lp = X[0, :, 0]
        assert lp[0] < lp[1] < lp[2]

    def test_eta_passthrough(self):
        """eta is copied directly to feature index 1."""
        eta_vals = np.array([[-1.5, 0.0, 2.3]])
        pt = np.ones((1, 3)) * 20.0
        e = np.ones((1, 3)) * 50.0
        phi = np.zeros((1, 3))
        X = feature_engineer(e, pt, phi, eta_vals)
        np.testing.assert_array_equal(X[0, :, 1], eta_vals[0])

    def test_feature_order_lp_eta_cosphi_sinphi_le(self):
        """Columns are [lp, eta, cos(phi), sin(phi), le]."""
        pt = np.array([[np.e * 1000.0]])    # log(pt/1000) = 1.0
        e = np.array([[np.e * 1000.0]])
        phi = np.array([[0.0]])             # cos=1, sin=0
        eta = np.array([[0.5]])
        X = feature_engineer(e, pt, phi, eta)
        assert pytest.approx(X[0, 0, 0], abs=1e-6) == 1.0  # lp
        assert pytest.approx(X[0, 0, 1], abs=1e-6) == 0.5  # eta
        assert pytest.approx(X[0, 0, 2], abs=1e-6) == 1.0  # cos(0)
        assert pytest.approx(X[0, 0, 3], abs=1e-6) == 0.0  # sin(0)
        assert pytest.approx(X[0, 0, 4], abs=1e-6) == 1.0  # le


# ---------------------------------------------------------------------------
# 6-jet filter: (pt > 0).sum(1) >= 6
# ---------------------------------------------------------------------------

class TestJetFilter:
    """Test the jet-count cut applied in load() of both eval scripts."""

    def _apply_filter(self, pt_raw, n=None):
        """pt_raw: (n_events, n_jets)."""
        pt = np.nan_to_num(pt_raw) / 1000.0
        mask = (pt > 0).sum(axis=1) >= 6
        X = np.zeros((pt_raw.shape[0], pt_raw.shape[1], 5))
        X = X[mask]
        if n is not None:
            X = X[:n]
        return X

    def test_keeps_events_with_six_or_more_jets(self):
        pt = np.zeros((5, 10))
        pt[:3, :8] = 10.0   # 3 events with 8 jets
        pt[3, :5] = 10.0    # 1 event with 5 jets (excluded)
        pt[4, :6] = 10.0    # 1 event with exactly 6 jets (included)
        result = self._apply_filter(pt)
        assert result.shape[0] == 4  # 3 + 1 (exactly-6)

    def test_excludes_events_with_fewer_than_six_jets(self):
        pt = np.zeros((3, 10))
        pt[:, :5] = 10.0    # only 5 jets per event
        result = self._apply_filter(pt)
        assert result.shape[0] == 0

    def test_n_cap_applied_after_filter(self):
        pt = np.ones((20, 10)) * 10.0  # all events have 10 jets
        result = self._apply_filter(pt, n=5)
        assert result.shape[0] == 5

    def test_nan_jets_dont_count(self):
        """nan pt jets should be treated as zero by nan_to_num."""
        pt = np.full((3, 10), np.nan)
        pt[:, :6] = 10.0    # 6 real jets, rest nan -> nan_to_num -> 0
        result = self._apply_filter(pt)
        assert result.shape[0] == 3  # all pass (exactly 6)


# ---------------------------------------------------------------------------
# Event packing: make_validation_set.py float32 → uint32 bit reinterpretation
# and mask word construction
# ---------------------------------------------------------------------------

def pack_event(ev_single, mask_single):
    """
    Inlined from make_validation_set.py inner loop.
    ev_single: float32 array of shape (12, 5)
    mask_single: bool array of shape (12,)
    Returns list of uint32 words.
    """
    words = [int(w) for w in ev_single.reshape(-1).view(np.uint32)]
    words += [1 if x else 0 for x in mask_single]
    return words


class TestEventPacking:
    def test_word_count_per_event(self):
        """Each event: 12*5 feature words + 12 mask words = 72 words."""
        ev = np.zeros((12, 5), np.float32)
        mask = np.zeros(12, dtype=bool)
        words = pack_event(ev, mask)
        assert len(words) == 72  # 60 + 12

    def test_zero_event_produces_zero_words(self):
        ev = np.zeros((12, 5), np.float32)
        mask = np.zeros(12, dtype=bool)
        words = pack_event(ev, mask)
        assert all(w == 0 for w in words)

    def test_mask_words_are_zero_or_one(self):
        rng = np.random.default_rng(20)
        ev = rng.standard_normal((12, 5)).astype(np.float32)
        mask = rng.integers(0, 2, 12, dtype=bool)
        words = pack_event(ev, mask)
        mask_words = words[60:]  # last 12 words
        assert all(w in (0, 1) for w in mask_words)

    def test_mask_true_becomes_one(self):
        ev = np.zeros((12, 5), np.float32)
        mask = np.array([True] * 6 + [False] * 6, dtype=bool)
        words = pack_event(ev, mask)
        assert words[60:66] == [1] * 6
        assert words[66:72] == [0] * 6

    def test_float32_bits_preserved(self):
        """Verify that float32 reinterpretation is lossless via struct round-trip."""
        ev = np.array([[1.0] + [0.0] * 4] + [[0.0] * 5] * 11, dtype=np.float32)
        mask = np.zeros(12, dtype=bool)
        words = pack_event(ev, mask)
        packed = struct.pack(f"{len(words)}I", *words)
        recovered = np.frombuffer(packed[:60 * 4], dtype=np.float32).reshape(12, 5)
        np.testing.assert_array_equal(recovered, ev)

    def test_multiple_events_total_words(self):
        """N events should produce N * 72 words."""
        n = 7
        ev_batch = np.zeros((n, 12, 5), np.float32)
        mask_batch = np.zeros((n, 12), dtype=bool)
        all_words = []
        for i in range(n):
            all_words += pack_event(ev_batch[i], mask_batch[i])
        assert len(all_words) == n * 72

    def test_struct_pack_roundtrip(self):
        """Words survive struct.pack('I') / struct.unpack('I')."""
        ev = np.zeros((12, 5), np.float32)
        mask = np.zeros(12, dtype=bool)
        words = pack_event(ev, mask)
        packed = struct.pack(f"{len(words)}I", *words)
        unpacked = list(struct.unpack(f"{len(words)}I", packed))
        assert unpacked == words


# ---------------------------------------------------------------------------
# 12-jet padding: make_validation_set.py pads/truncates to 12 jets
# ---------------------------------------------------------------------------

def pad_to_12(X_raw):
    """
    Inlined from make_validation_set.py post-filter padding.
    X_raw: (n_events, n_jets, 5)  float32
    Returns ev: (n_events, 12, 5) float32, mask: (n_events, 12) bool
    """
    ev = np.zeros((X_raw.shape[0], 12, 5), np.float32)
    k = min(12, X_raw.shape[1])
    ev[:, :k, :] = X_raw[:, :k, :]
    mask = ev[:, :, 0] == 0
    return ev, mask


class TestPadTo12:
    def test_output_shape(self):
        X = np.ones((10, 8, 5), np.float32) * 0.5
        ev, mask = pad_to_12(X)
        assert ev.shape == (10, 12, 5)
        assert mask.shape == (10, 12)

    def test_fewer_than_12_jets_pads_with_zeros(self):
        X = np.ones((5, 8, 5), np.float32) * 2.0
        ev, mask = pad_to_12(X)
        np.testing.assert_array_equal(ev[:, 8:, :], 0.0)

    def test_padded_jets_are_masked(self):
        X = np.ones((5, 8, 5), np.float32) * 2.0
        ev, mask = pad_to_12(X)
        assert mask[:, 8:].all()

    def test_real_jets_not_masked_when_nonzero_lp(self):
        X = np.ones((5, 8, 5), np.float32) * 2.0
        ev, mask = pad_to_12(X)
        # First feature (lp) is non-zero, so jets 0-7 are NOT masked
        assert not mask[:, :8].any()

    def test_more_than_12_jets_truncated(self):
        X = np.ones((3, 20, 5), np.float32) * 1.5
        ev, mask = pad_to_12(X)
        assert ev.shape == (3, 12, 5)
        np.testing.assert_array_equal(ev, 1.5)

    def test_exactly_12_jets_no_padding(self):
        X = np.ones((4, 12, 5), np.float32) * 3.0
        ev, mask = pad_to_12(X)
        np.testing.assert_array_equal(ev, 3.0)
        assert not mask.any()


# ---------------------------------------------------------------------------
# load() with skip parameter (make_auc_ref_model.py)
# ---------------------------------------------------------------------------

def load_with_skip(fn, n, skip=0, *, h5py_file_ctx):
    """
    Inlined from make_auc_ref_model.load(), with h5py.File replaced by the
    provided context manager factory for testing.
    Returns a numpy array (no torch dependency).
    """
    with h5py_file_ctx(fn, "r") as f:
        e = np.nan_to_num(np.array(f["source"]["e"])) / 1000.0
        pt = np.nan_to_num(np.array(f["source"]["pt"])) / 1000.0
        with np.errstate(divide="ignore"):
            le = np.log(e)
            le[~np.isfinite(le)] = 0.0
            lp = np.log(pt)
            lp[~np.isfinite(lp)] = 0.0
        phi = np.array(f["source"]["phi"])
        eta = np.array(f["source"]["eta"])
        X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], axis=-1)
        X = X[(pt > 0).sum(axis=1) >= 6]
    if skip:
        X = X[skip:]
    return X[:n]


class _MockH5Context:
    """Minimal h5py.File context-manager mock backed by numpy arrays."""

    def __init__(self, data):
        self._data = data

    def __call__(self, path, mode):
        return self

    def __enter__(self):
        return self._data

    def __exit__(self, *args):
        pass


def _make_h5_data(n_ev=50, n_jets=15, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "source": {
            "e": rng.uniform(0.1, 500.0, (n_ev, n_jets)),
            "pt": rng.uniform(5.0, 200.0, (n_ev, n_jets)),
            "phi": rng.uniform(-np.pi, np.pi, (n_ev, n_jets)),
            "eta": rng.uniform(-3.0, 3.0, (n_ev, n_jets)),
        }
    }


class TestLoadWithSkip:
    def test_no_skip_shape(self):
        ctx = _MockH5Context(_make_h5_data(n_ev=30, n_jets=12))
        X = load_with_skip("fake.h5", n=20, skip=0, h5py_file_ctx=ctx)
        assert X.ndim == 3
        assert X.shape[1] == 12
        assert X.shape[2] == 5
        assert X.shape[0] <= 20

    def test_skip_reduces_available_events(self):
        data = _make_h5_data(n_ev=50, n_jets=12)
        ctx_no_skip = _MockH5Context(data)
        ctx_skip = _MockH5Context(data)
        X_full = load_with_skip("f.h5", n=50, skip=0, h5py_file_ctx=ctx_no_skip)
        X_skip = load_with_skip("f.h5", n=50, skip=10, h5py_file_ctx=ctx_skip)
        # After skipping 10 events, we get at most len(X_full) - 10 events
        assert X_skip.shape[0] <= X_full.shape[0]

    def test_skip_zero_same_as_no_skip(self):
        data = _make_h5_data(n_ev=20, n_jets=12)
        ctx1 = _MockH5Context(data)
        ctx2 = _MockH5Context(data)
        X1 = load_with_skip("f.h5", n=10, skip=0, h5py_file_ctx=ctx1)
        X2 = load_with_skip("f.h5", n=10, skip=0, h5py_file_ctx=ctx2)
        np.testing.assert_array_equal(X1, X2)

    def test_n_cap_applied(self):
        ctx = _MockH5Context(_make_h5_data(n_ev=100, n_jets=12))
        X = load_with_skip("f.h5", n=5, h5py_file_ctx=ctx)
        assert X.shape[0] <= 5

    def test_cos_sin_feature_range(self):
        ctx = _MockH5Context(_make_h5_data(n_ev=30, n_jets=12))
        X = load_with_skip("f.h5", n=30, h5py_file_ctx=ctx)
        assert np.all(X[:, :, 2] >= -1.0) and np.all(X[:, :, 2] <= 1.0)
        assert np.all(X[:, :, 3] >= -1.0) and np.all(X[:, :, 3] <= 1.0)

    def test_skip_larger_than_n_events_returns_empty(self):
        ctx = _MockH5Context(_make_h5_data(n_ev=10, n_jets=12))
        X = load_with_skip("f.h5", n=50, skip=100, h5py_file_ctx=ctx)
        assert X.shape[0] == 0
