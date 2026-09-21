"""Tests for data packing / byte-order / C-header generation.

Scripts covered:
  src/pl_stream/pack_input.py               – float_to_ap_fixed()
  src/pl_stream/export_weights_to_header.py – to_c(), shape_str(),
      emit_float_array(), emit_copy()
  scripts/pack_eval_float.py                – load() (via mock h5py)

Both pl_stream scripts have __main__ guards and are imported safely via
importlib.  pack_eval_float.py reads an HDF5 file and a sys.argv argument at
module level; its load() function is inlined and tested with a mocked h5py.
"""
import importlib.util
import io
import os
import struct
import sys
import unittest.mock as mock

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
PL_STREAM_DIR = os.path.join(SRC_DIR, "pl_stream")
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")

_H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None


# ===========================================================================
# pack_input.float_to_ap_fixed
# ap_fixed<16,7>: 9 fractional bits, range roughly ±64, floor towards -inf
# ===========================================================================

@pytest.fixture(scope="module")
def pack_input_mod():
    spec = importlib.util.spec_from_file_location(
        "pack_input", os.path.join(PL_STREAM_DIR, "pack_input.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFloatToApFixed:
    def test_zero(self, pack_input_mod):
        assert pack_input_mod.float_to_ap_fixed(0.0) == 0

    def test_positive_one(self, pack_input_mod):
        # 1.0 * 2^9 = 512
        assert pack_input_mod.float_to_ap_fixed(1.0) == 512

    def test_half(self, pack_input_mod):
        # 0.5 * 2^9 = 256
        assert pack_input_mod.float_to_ap_fixed(0.5) == 256

    def test_negative_one_twos_complement(self, pack_input_mod):
        # -1.0 * 512 = -512 -> 0xFE00 = 65024
        result = pack_input_mod.float_to_ap_fixed(-1.0)
        assert result == ((-512) & 0xFFFF)

    def test_floor_not_round(self, pack_input_mod):
        """AP_TRN truncates towards -inf, not rounds."""
        # 1.9 * 512 = 972.8 -> floor = 972
        expected = int(np.floor(1.9 * 512)) & 0xFFFF
        assert pack_input_mod.float_to_ap_fixed(1.9) == expected

    def test_clamped_high(self, pack_input_mod):
        max_val = (2 ** 15 - 1) / (2 ** 9)
        r1 = pack_input_mod.float_to_ap_fixed(max_val)
        r2 = pack_input_mod.float_to_ap_fixed(max_val * 1000)
        assert r1 == r2

    def test_clamped_low(self, pack_input_mod):
        min_val = -(2 ** 15) / (2 ** 9)
        r1 = pack_input_mod.float_to_ap_fixed(min_val)
        r2 = pack_input_mod.float_to_ap_fixed(min_val * 1000)
        assert r1 == r2

    def test_result_fits_uint16(self, pack_input_mod):
        for v in [-100.0, -1.0, 0.0, 0.5, 1.0, 63.0]:
            result = pack_input_mod.float_to_ap_fixed(v)
            assert 0 <= result <= 0xFFFF

    def test_negative_half_twos_complement(self, pack_input_mod):
        # -0.5 * 512 = -256 -> 0xFF00 = 65280
        result = pack_input_mod.float_to_ap_fixed(-0.5)
        assert result == ((-256) & 0xFFFF)

    def test_small_positive(self, pack_input_mod):
        # 0.001 * 512 = 0.512 -> floor = 0
        assert pack_input_mod.float_to_ap_fixed(0.001) == 0

    def test_two_frac(self, pack_input_mod):
        # exact power of two: 2.0 * 512 = 1024
        assert pack_input_mod.float_to_ap_fixed(2.0) == 1024


# ===========================================================================
# export_weights_to_header: to_c, shape_str, emit_float_array, emit_copy
# ===========================================================================

@pytest.fixture(scope="module")
def ewth_mod():
    spec = importlib.util.spec_from_file_location(
        "export_weights_to_header",
        os.path.join(PL_STREAM_DIR, "export_weights_to_header.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestShapeStr:
    def test_1d(self, ewth_mod):
        assert ewth_mod.shape_str(np.zeros(8)) == "[8]"

    def test_2d(self, ewth_mod):
        assert ewth_mod.shape_str(np.zeros((4, 16))) == "[4][16]"

    def test_3d(self, ewth_mod):
        assert ewth_mod.shape_str(np.zeros((2, 3, 4))) == "[2][3][4]"

    def test_single_element(self, ewth_mod):
        assert ewth_mod.shape_str(np.zeros((1,))) == "[1]"


class TestToC:
    def test_starts_with_brace(self, ewth_mod):
        s = ewth_mod.to_c(np.array([1.0, 2.0]))
        assert s.startswith("{")

    def test_ends_with_brace(self, ewth_mod):
        s = ewth_mod.to_c(np.array([1.0]))
        assert s.endswith("}")

    def test_2d_nested_braces(self, ewth_mod):
        s = ewth_mod.to_c(np.ones((2, 3)))
        assert "{{" in s

    def test_float_formatting(self, ewth_mod):
        s = ewth_mod.to_c(np.array([0.5]))
        assert "0.5" in s

    def test_negative_value(self, ewth_mod):
        s = ewth_mod.to_c(np.array([-1.0]))
        assert "-1" in s


class TestEmitFloatArray:
    def test_static_const_float_decl(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_float_array(buf, "test_arr", np.array([1.0, 2.0, 3.0]))
        out = buf.getvalue()
        assert "static const float test_arr[3]" in out

    def test_contains_value(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_float_array(buf, "w", np.array([3.14]))
        assert "3.14" in buf.getvalue()

    def test_ends_with_semicolon(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_float_array(buf, "w", np.zeros(4))
        assert buf.getvalue().rstrip().endswith(";")


class TestEmitCopy:
    def test_1d_single_loop(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_copy(buf, "dst", "src", np.zeros(4))
        out = buf.getvalue()
        assert out.count("for") == 1
        assert "4" in out
        assert "dst" in out and "src" in out

    def test_2d_two_loops(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_copy(buf, "d", "s", np.zeros((3, 5)))
        out = buf.getvalue()
        assert out.count("for") == 2
        assert "3" in out
        assert "5" in out

    def test_3d_three_loops(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_copy(buf, "d", "s", np.zeros((2, 3, 4)))
        out = buf.getvalue()
        assert out.count("for") == 3

    def test_assignment_in_output(self, ewth_mod):
        buf = io.StringIO()
        ewth_mod.emit_copy(buf, "x", "y", np.zeros(6))
        assert "x" in buf.getvalue()
        assert "y" in buf.getvalue()
        assert "=" in buf.getvalue()


# ===========================================================================
# pack_eval_float.load (inlined; h5py mocked)
# ===========================================================================

def load_eval(fn, n):
    """Inlined from scripts/pack_eval_float.load, h5py.File injected via mock."""
    import h5py
    with h5py.File(fn, "r") as f:
        e = np.nan_to_num(np.array(f["source"]["e"])) / 1000.0
        pt = np.nan_to_num(np.array(f["source"]["pt"])) / 1000.0
        with np.errstate(divide="ignore"):
            le = np.log(e)
            le[~np.isfinite(le)] = 0
            lp = np.log(pt)
            lp[~np.isfinite(lp)] = 0
        phi = np.array(f["source"]["phi"])
        eta = np.array(f["source"]["eta"])
        X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], -1)
        X = X[(pt > 0).sum(1) >= 6][:n]
    out = np.zeros((X.shape[0], 12, 5), np.float32)
    k = min(12, X.shape[1])
    out[:, :k, :] = X[:, :k, :]
    mask = out[:, :, 0] == 0
    return out, mask


def _make_h5_context(n_ev=50, n_jets=20, seed=99):
    rng = np.random.default_rng(seed)
    data = {
        "source": {
            "e": rng.uniform(0.1, 500.0, (n_ev, n_jets)),
            "pt": rng.uniform(5.0, 200.0, (n_ev, n_jets)),
            "phi": rng.uniform(-np.pi, np.pi, (n_ev, n_jets)),
            "eta": rng.uniform(-3.0, 3.0, (n_ev, n_jets)),
        }
    }

    class _CM:
        def __init__(self, d):
            self.d = d
        def __enter__(self):
            return self.d
        def __exit__(self, *a):
            pass

    return _CM(data)


@pytest.mark.skipif(not _H5PY_AVAILABLE, reason="h5py not installed")
class TestPackEvalFloat:
    def test_output_shape(self):
        ctx = _make_h5_context(n_ev=50, n_jets=20)
        with mock.patch("h5py.File", return_value=ctx):
            X, mask = load_eval("fake.h5", 30)
        assert X.shape[0] <= 30
        assert X.shape[1] == 12
        assert X.shape[2] == 5

    def test_mask_shape(self):
        ctx = _make_h5_context(n_ev=50, n_jets=20)
        with mock.patch("h5py.File", return_value=ctx):
            X, mask = load_eval("fake.h5", 20)
        assert mask.shape == X.shape[:2]

    def test_padded_jets_masked(self):
        """With only 8 real jets, columns 8-11 should be masked (pt=0 -> lp=0)."""
        ctx = _make_h5_context(n_ev=10, n_jets=8)
        with mock.patch("h5py.File", return_value=ctx):
            X, mask = load_eval("fake.h5", 10)
        assert mask[:, 8:].all()

    def test_n_events_cap(self):
        ctx = _make_h5_context(n_ev=20, n_jets=20)
        with mock.patch("h5py.File", return_value=ctx):
            X, _ = load_eval("fake.h5", 10)
        assert X.shape[0] <= 10

    def test_dtype_float32(self):
        ctx = _make_h5_context()
        with mock.patch("h5py.File", return_value=ctx):
            X, _ = load_eval("fake.h5", 5)
        assert X.dtype == np.float32

    def test_cos_sin_range(self):
        """Features 2 and 3 are cos(phi) and sin(phi): must be in [-1, 1]."""
        ctx = _make_h5_context(n_ev=30, n_jets=15)
        with mock.patch("h5py.File", return_value=ctx):
            X, _ = load_eval("fake.h5", 30)
        real = ~(X[:, :, 0] == 0)  # non-padded jets
        cos_vals = X[:, :, 2][real]
        sin_vals = X[:, :, 3][real]
        assert np.all(cos_vals >= -1.0) and np.all(cos_vals <= 1.0)
        assert np.all(sin_vals >= -1.0) and np.all(sin_vals <= 1.0)


# ===========================================================================
# Struct packing round-trip
# ===========================================================================

class TestStructPackRoundtrip:
    """Verify that ap_fixed words survive struct.pack('I') / struct.unpack('I')."""

    FRAC = 9
    SCALE = 2.0 ** FRAC

    def _encode(self, v):
        max_v = (2 ** 15 - 1) / self.SCALE
        min_v = -(2 ** 15) / self.SCALE
        v = float(np.clip(v, min_v, max_v))
        return int(np.floor(v * self.SCALE)) & 0xFFFF

    def test_zero(self):
        words = [self._encode(0.0)]
        packed = struct.pack(f"{len(words)}I", *words)
        assert struct.unpack(f"{len(words)}I", packed) == tuple(words)

    def test_roundtrip_multiple(self):
        values = [0.5, -1.0, 0.0, 1.9, -3.5, 10.0]
        words = [self._encode(v) for v in values]
        packed = struct.pack(f"{len(words)}I", *words)
        assert list(struct.unpack(f"{len(words)}I", packed)) == words

    def test_event_word_count(self):
        """Each event packs 12*5 feature words + 12 mask words = 72 words."""
        n_ev = 3
        words_per_event = 12 * 5 + 12   # = 72
        words = [0] * (n_ev * words_per_event)
        packed = struct.pack(f"{len(words)}I", *words)
        assert len(packed) == n_ev * words_per_event * 4  # 4 bytes each
