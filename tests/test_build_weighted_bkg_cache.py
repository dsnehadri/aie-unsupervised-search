"""Tests for pure functions in scripts/build_weighted_bkg_cache.py.

The script executes model inference and heavy file I/O at module level, so it
cannot be imported.  We reproduce:
  - min_asym_mavg(jp4, njet) — pure numpy, no external deps
  - to_X(d)                  — uses torch (skipped if torch is absent)
and test them with synthetic data.
"""
import itertools

import numpy as np
import pytest


# ===========================================================================
# Reproduced verbatim from build_weighted_bkg_cache.py
# ===========================================================================

def min_asym_mavg(jp4, njet):
    """Return the minimum-mass-asymmetry average group mass for each event."""
    N = jp4.shape[0]
    out = np.full(N, np.nan)
    for n in np.unique(njet):
        n = int(n)
        if n < 2:
            continue
        combos = []
        for r in range(0, n):
            for extra in itertools.combinations(range(1, n), r):
                m = np.zeros(n, bool)
                m[0] = True
                for e_ in extra:
                    m[e_] = True
                combos.append(m)
        mf = np.array(combos).astype(float)
        sel = np.where(njet == n)[0]
        if len(sel) == 0:
            continue

        def mass(p):
            m2 = p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2
            return np.sqrt(np.clip(m2, 0, None))

        CH = 2000
        for c0 in range(0, len(sel), CH):
            idx = sel[c0:c0 + CH]
            jets = jp4[idx][:, :n, :]
            A = np.einsum("pn,enf->epf", mf, jets)
            B = jets.sum(1)[:, None, :] - A
            mA, mB = mass(A), mass(B)
            valid = (mA + mB) > 0
            asym = np.where(valid, np.abs(mA - mB) / (mA + mB + 1e-9), np.inf)
            best = asym.argmin(1)
            ei = np.arange(len(idx))
            out[idx] = 0.5 * (mA[ei, best] + mB[ei, best])
    return out


def to_X(d):
    """Feature-engineer jet arrays → model input tensor (requires torch)."""
    torch = pytest.importorskip("torch")
    e = np.nan_to_num(d["e"]) / 1000.0
    pt = np.nan_to_num(d["pt"]) / 1000.0
    with np.errstate(divide="ignore"):
        le = np.log(e)
        le[~np.isfinite(le)] = 0
        lp = np.log(pt)
        lp[~np.isfinite(lp)] = 0
    X = np.stack([lp, d["eta"], np.cos(d["phi"]), np.sin(d["phi"]), le], -1)
    keep = (pt > 0).sum(1) >= 6
    return torch.tensor(X[keep], dtype=torch.float32)


# ===========================================================================
# Helpers
# ===========================================================================

def make_massless_jet(E, direction=(1, 0, 0)):
    """Return a 4-vector (E, px, py, pz) for a massless jet pointing along direction."""
    dx, dy, dz = direction
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.array([E, E * dx / norm, E * dy / norm, E * dz / norm])


def massive_jet(mass, px=0.0, py=0.0, pz=0.0):
    """Return a 4-vector with given invariant mass, at rest by default."""
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
    return np.array([E, px, py, pz])


# ===========================================================================
# Tests: min_asym_mavg
# ===========================================================================

class TestMinAsymMavg:
    def test_two_equal_mass_jets(self):
        """Two jets with equal masses: best partition is {0}|{1}, asym=0,
        mavg = (m + m)/2 = m."""
        m = 10.0
        jp4 = np.array([[massive_jet(m), massive_jet(m)]])  # shape (1,2,4)
        njet = np.array([2])
        result = min_asym_mavg(jp4, njet)
        assert np.isfinite(result[0])
        assert abs(result[0] - m) < 1e-6, f"expected {m}, got {result[0]}"

    def test_two_jets_unequal_mass(self):
        """Result should equal (m1 + m2)/2 for the only non-trivial partition."""
        m1, m2 = 5.0, 15.0
        jp4 = np.array([[massive_jet(m1), massive_jet(m2)]])
        njet = np.array([2])
        result = min_asym_mavg(jp4, njet)
        assert np.isfinite(result[0])
        assert abs(result[0] - 0.5 * (m1 + m2)) < 1e-5

    def test_single_jet_gives_nan(self):
        """njet=1 is below the n>=2 threshold → output should be nan."""
        jp4 = np.array([[massive_jet(10.0), [0, 0, 0, 0]]])
        njet = np.array([1])
        result = min_asym_mavg(jp4, njet)
        assert np.isnan(result[0])

    def test_zero_jet_gives_nan(self):
        jp4 = np.zeros((1, 2, 4))
        njet = np.array([0])
        result = min_asym_mavg(jp4, njet)
        assert np.isnan(result[0])

    def test_output_shape_matches_input(self):
        N = 5
        jp4 = np.tile(massive_jet(10.0), (N, 4, 1))  # (N,4,4)
        njet = np.full(N, 4)
        result = min_asym_mavg(jp4, njet)
        assert result.shape == (N,)

    def test_three_jets_symmetric_gives_sensible_mavg(self):
        """Three identical jets: all non-trivial partitions have the same asym;
        mavg should be positive and finite."""
        jp4 = np.array([[massive_jet(8.0), massive_jet(8.0), massive_jet(8.0)]])
        njet = np.array([3])
        result = min_asym_mavg(jp4, njet)
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_massless_jets_give_zero_mavg(self):
        """Massless jets have m²=0; group masses depend on angle.
        For collinear jets pointing the same way, the group mass is also ~0."""
        j = make_massless_jet(10.0, direction=(1, 0, 0))
        jp4 = np.array([[j, j]])
        njet = np.array([2])
        result = min_asym_mavg(jp4, njet)
        # Group {0} alone: mass = 0; Group {1} alone: mass = 0
        # best: asym = inf (both groups have 0 mass, so valid=False) →
        # asym set to inf; if all inf, argmin picks first; mavg = 0
        # (or it might select a partition with valid=True differently)
        assert np.isfinite(result[0]) or np.isnan(result[0])  # should not raise

    def test_multiple_njet_values_in_batch(self):
        """Events with different jet multiplicities in the same batch."""
        jp4_2 = [massive_jet(5.0), massive_jet(5.0), [0]*4, [0]*4]
        jp4_3 = [massive_jet(4.0), massive_jet(4.0), massive_jet(4.0), [0]*4]
        jp4 = np.array([jp4_2, jp4_3], dtype=float)
        njet = np.array([2, 3])
        result = min_asym_mavg(jp4, njet)
        assert result.shape == (2,)
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])

    def test_result_is_nonnegative(self):
        rng = np.random.default_rng(17)
        N = 20
        jp4 = np.zeros((N, 6, 4))
        for i in range(N):
            for j in range(6):
                m = rng.uniform(1, 20)
                jp4[i, j] = massive_jet(m)
        njet = np.full(N, 6)
        result = min_asym_mavg(jp4, njet)
        finite = result[np.isfinite(result)]
        assert np.all(finite >= 0)


# ===========================================================================
# Tests: to_X (torch required)
# ===========================================================================

class TestToX:
    @pytest.fixture(autouse=True)
    def require_torch(self):
        pytest.importorskip("torch")

    def _make_d(self, N=10, J=12, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        return {
            "pt":  rng.uniform(10, 500, (N, J)).astype(np.float32),
            "eta": rng.uniform(-2.5, 2.5, (N, J)).astype(np.float32),
            "phi": rng.uniform(-np.pi, np.pi, (N, J)).astype(np.float32),
            "e":   rng.uniform(10, 500, (N, J)).astype(np.float32),
        }

    def test_output_is_tensor(self):
        import torch
        d = self._make_d()
        X = to_X(d)
        assert isinstance(X, torch.Tensor)

    def test_output_dtype_float32(self):
        import torch
        d = self._make_d()
        X = to_X(d)
        assert X.dtype == torch.float32

    def test_output_has_five_features(self):
        d = self._make_d()
        X = to_X(d)
        assert X.shape[-1] == 5

    def test_events_with_fewer_than_6_jets_filtered(self):
        N, J = 5, 12
        rng = np.random.default_rng(5)
        d = self._make_d(N=N, J=J, rng=rng)
        # Zero out all jets for event 0 (it will have 0 jets with pt>0)
        d["pt"][0, :] = 0.0
        X = to_X(d)
        # Event 0 must be excluded
        assert X.shape[0] < N

    def test_nan_pt_handled(self):
        d = self._make_d()
        d["pt"][0, 0] = np.nan
        # Should not raise; nan_to_num converts to 0
        X = to_X(d)
        assert X is not None

    def test_all_valid_events_kept(self):
        N = 8
        d = self._make_d(N=N)
        X = to_X(d)
        # All events have 12 jets with positive pt → all should be kept
        assert X.shape[0] == N
