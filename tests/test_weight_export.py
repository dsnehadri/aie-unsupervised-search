"""Tests for weight quantization and C-header emission.

Scripts covered:
  scripts/export_embed_weights_aie.py    – q(), emit(), K padding
  scripts/export_pairwise_weights_aie.py – q(), emit(), K/N padding
  src/attn_block_aie/slice_weights_for_aie.py – to_fixed16(), format_array(),
      slice_per_head(), slice_bias_per_head(), slice_bias_kv_per_head()

The export scripts load .npy checkpoint files and write a real output header;
they execute all their loading code at module level, so we cannot import them
safely without the weight files.  Instead we inline the pure functions and
test those.  slice_weights_for_aie.py has a __main__ guard and is imported via
importlib.
"""
import importlib.util
import io
import os
import sys

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
AIE_DIR = os.path.join(SRC_DIR, "attn_block_aie")


# ===========================================================================
# q() — fixed-point quantization (identical in both export scripts)
# ===========================================================================

def q(a, frac):
    """Inlined from export_embed_weights_aie / export_pairwise_weights_aie."""
    v = np.rint(np.asarray(a, np.float64) * (1 << frac))
    assert np.abs(v).max() <= 32767, (
        f"overflow at Q.{frac}: max |a| = {np.abs(a).max():.4f}"
    )
    return v.astype(np.int16)


class TestQuantization:
    def test_dtype_is_int16(self):
        a = np.array([0.5, -0.25, 0.125])
        assert q(a, frac=9).dtype == np.int16

    def test_zero_maps_to_zero(self):
        np.testing.assert_array_equal(q(np.zeros(8), frac=12), 0)

    def test_known_value_data_scale(self):
        # 0.5 * 2^9 = 256
        assert q(np.array([0.5]), frac=9)[0] == 256

    def test_known_value_weight_scale(self):
        # 0.25 * 2^12 = 1024
        assert q(np.array([0.25]), frac=12)[0] == 1024

    def test_negative_value(self):
        # -1.0 * 2^9 = -512
        assert q(np.array([-1.0]), frac=9)[0] == -512

    def test_rounding_is_nearest(self):
        # 0.001 * 2^12 = 4.096 -> rounds to 4
        expected = int(np.rint(0.001 * (1 << 12)))
        assert q(np.array([0.001]), frac=12)[0] == expected

    def test_overflow_raises(self):
        # 100.0 * 2^12 = 409600 >> 32767
        with pytest.raises(AssertionError, match="overflow"):
            q(np.array([100.0]), frac=12)

    def test_shape_preserved(self):
        a = np.arange(16, dtype=np.float64) * 0.001
        assert q(a, frac=9).shape == a.shape

    def test_2d_shape_preserved(self):
        a = np.ones((4, 8), dtype=np.float64) * 0.1
        assert q(a, frac=9).shape == (4, 8)

    def test_values_in_int16_range(self):
        a = np.linspace(-0.5, 0.5, 100)
        result = q(a, frac=9)
        assert result.min() >= -32768
        assert result.max() <= 32767


# ===========================================================================
# emit() — write int16 C-array to a file-like object
# ===========================================================================

def emit(f, name, arr, frac):
    """Inlined from export_embed_weights_aie.emit."""
    v = q(arr, frac).ravel()
    f.write(f"alignas(16) static const int16 {name}[{v.size}] = {{\n")
    for i in range(0, v.size, 16):
        f.write("    " + ", ".join(f"{x:6d}" for x in v[i:i + 16]) + ",\n")
    f.write("};\n\n")


class TestEmit:
    def test_header_line_contains_name(self):
        buf = io.StringIO()
        emit(buf, "my_arr", np.zeros(4), frac=9)
        assert "alignas(16) static const int16 my_arr[4]" in buf.getvalue()

    def test_correct_element_count_in_header(self):
        buf = io.StringIO()
        emit(buf, "w", np.ones(32) * 0.5, frac=9)
        assert "w[32]" in buf.getvalue()

    def test_known_values_appear_in_output(self):
        buf = io.StringIO()
        # 0.5*512=256, -0.5*512=-256
        emit(buf, "t", np.array([0.5, -0.5, 0.0]), frac=9)
        out = buf.getvalue()
        assert "256" in out
        assert "-256" in out

    def test_output_ends_with_brace(self):
        buf = io.StringIO()
        emit(buf, "a", np.zeros(8), frac=9)
        assert buf.getvalue().strip().endswith("};")

    def test_2d_array_flattened(self):
        buf = io.StringIO()
        arr = np.ones((4, 4)) * 0.25
        emit(buf, "mat", arr, frac=9)
        # 4*4 = 16 elements
        assert "mat[16]" in buf.getvalue()

    def test_lines_of_16(self):
        buf = io.StringIO()
        emit(buf, "x", np.ones(32) * 0.1, frac=9)
        out = buf.getvalue()
        # Lines between the braces should have at most 16 values each
        for line in out.splitlines():
            line = line.strip().rstrip(",")
            if line and line[0].lstrip("-").isdigit():
                n_vals = len(line.split(","))
                assert n_vals <= 16


# ===========================================================================
# embed weight padding: K padded 5 -> 8
# ===========================================================================

class TestEmbedWeightPadding:
    def test_W0_shape(self):
        K0P, E = 8, 16
        W0 = np.random.default_rng(0).normal(0, 0.1, (E, 5))
        W0p = np.zeros((K0P, E), np.float64)
        W0p[:W0.shape[1], :] = W0.T
        assert W0p.shape == (K0P, E)

    def test_W0_padding_rows_are_zero(self):
        K0P, E = 8, 16
        W0 = np.random.default_rng(0).normal(0, 0.1, (E, 5))
        W0p = np.zeros((K0P, E), np.float64)
        W0p[:W0.shape[1], :] = W0.T
        np.testing.assert_array_equal(W0p[5:, :], 0)

    def test_W0_data_rows_preserved(self):
        K0P, E = 8, 16
        W0 = np.random.default_rng(1).normal(0, 0.1, (E, 5))
        W0p = np.zeros((K0P, E), np.float64)
        W0p[:W0.shape[1], :] = W0.T
        np.testing.assert_array_almost_equal(W0p[:5, :], W0.T)


# ===========================================================================
# pairwise weight padding: K padded 3->4, N padded 1->8
# ===========================================================================

class TestPairwiseWeightPadding:
    def test_W0_shape(self):
        H = 16
        W0 = np.random.default_rng(0).normal(0, 0.1, (H, 3))
        W0p = np.zeros((4, H))
        W0p[:3, :] = W0.T
        assert W0p.shape == (4, H)

    def test_W0_padding_zero(self):
        H = 16
        W0 = np.random.default_rng(0).normal(0, 0.1, (H, 3))
        W0p = np.zeros((4, H))
        W0p[:3, :] = W0.T
        np.testing.assert_array_equal(W0p[3:, :], 0)

    def test_W9_shape(self):
        H = 16
        W9 = np.random.default_rng(1).normal(0, 0.1, (1, H))
        W9p = np.zeros((H, 8))
        W9p[:, 0] = W9[0]
        assert W9p.shape == (H, 8)

    def test_W9_data_column(self):
        H = 16
        W9 = np.random.default_rng(2).normal(0, 0.1, (1, H))
        W9p = np.zeros((H, 8))
        W9p[:, 0] = W9[0]
        np.testing.assert_array_almost_equal(W9p[:, 0], W9[0])

    def test_W9_padding_columns_zero(self):
        H = 16
        W9 = np.random.default_rng(2).normal(0, 0.1, (1, H))
        W9p = np.zeros((H, 8))
        W9p[:, 0] = W9[0]
        np.testing.assert_array_equal(W9p[:, 1:], 0)

    def test_b9_padding(self):
        b9 = np.array([0.123])
        b9p = np.zeros(8)
        b9p[0] = b9[0]
        assert b9p[0] == pytest.approx(0.123)
        np.testing.assert_array_equal(b9p[1:], 0)


# ===========================================================================
# slice_weights_for_aie.py – import via importlib (safe: has __main__ guard)
# ===========================================================================

@pytest.fixture(scope="module")
def slice_mod():
    spec = importlib.util.spec_from_file_location(
        "slice_weights_for_aie",
        os.path.join(AIE_DIR, "slice_weights_for_aie.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestToFixed16:
    def test_dtype_int16(self, slice_mod):
        result = slice_mod.to_fixed16(np.array([0.5, -0.5]), frac_bits=11)
        assert result.dtype == np.int16

    def test_known_value(self, slice_mod):
        # 0.5 * 2^11 = 1024
        assert slice_mod.to_fixed16(np.array([0.5]), frac_bits=11)[0] == 1024

    def test_zero(self, slice_mod):
        assert slice_mod.to_fixed16(np.array([0.0]), frac_bits=11)[0] == 0

    def test_clamp_positive(self, slice_mod):
        # Very large value should be clipped to 32767
        assert slice_mod.to_fixed16(np.array([1e6]), frac_bits=11)[0] == 32767

    def test_clamp_negative(self, slice_mod):
        assert slice_mod.to_fixed16(np.array([-1e6]), frac_bits=11)[0] == -32768

    def test_shape_preserved(self, slice_mod):
        a = np.ones((3, 4), dtype=np.float64)
        result = slice_mod.to_fixed16(a, frac_bits=9)
        assert result.shape == (3, 4)


class TestFormatArray:
    def test_contains_name(self, slice_mod):
        a = np.zeros(8, dtype=np.int16)
        out = slice_mod.format_array("my_weight", a)
        assert "my_weight" in out

    def test_size_in_header(self, slice_mod):
        a = np.ones(16, dtype=np.int16)
        out = slice_mod.format_array("w", a)
        assert "w[16]" in out

    def test_alignas_present(self, slice_mod):
        a = np.zeros(4, dtype=np.int16)
        out = slice_mod.format_array("x", a)
        assert "alignas(16)" in out

    def test_closes_with_semicolon(self, slice_mod):
        a = np.zeros(4, dtype=np.int16)
        out = slice_mod.format_array("x", a)
        assert out.rstrip().endswith("};")


class TestSlicePerHead:
    def test_shape_E_d(self, slice_mod):
        E = slice_mod.E_DIM   # 16
        d = slice_mod.D_HEAD  # 4
        full_w = np.random.default_rng(0).normal(0, 0.1, (3 * E, E))
        for h in range(slice_mod.N_HEADS):
            Wq, Wk, Wv = slice_mod.slice_per_head(full_w, h)
            assert Wq.shape == (E, d)
            assert Wk.shape == (E, d)
            assert Wv.shape == (E, d)

    def test_correct_rows_extracted(self, slice_mod):
        """Wq for head h comes from rows [h*d:(h+1)*d] of the Q block."""
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        full_w = np.eye(3 * E, E)
        h = 1
        Wq, _, _ = slice_mod.slice_per_head(full_w, h)
        expected = full_w[h * d:(h + 1) * d, :].T
        np.testing.assert_array_equal(Wq, expected)

    def test_all_heads_tile_to_full_Wq(self, slice_mod):
        """Concatenating Wq^T over heads should recover the Q block of in_proj."""
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        rng = np.random.default_rng(5)
        full_w = rng.normal(0, 1, (3 * E, E))
        Wq_rows = np.concatenate(
            [slice_mod.slice_per_head(full_w, h)[0].T for h in range(slice_mod.N_HEADS)],
            axis=0,
        )  # (E, E)
        np.testing.assert_array_equal(Wq_rows, full_w[:E, :])


class TestSliceBiasPerHead:
    def test_shape(self, slice_mod):
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        full_b = np.arange(3 * E, dtype=np.float32)
        for h in range(slice_mod.N_HEADS):
            bq, bk, bv = slice_mod.slice_bias_per_head(full_b, h)
            assert bq.shape == (d,)
            assert bk.shape == (d,)
            assert bv.shape == (d,)

    def test_values_bq(self, slice_mod):
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        full_b = np.arange(3 * E, dtype=np.float64)
        h = 0
        bq, _, _ = slice_mod.slice_bias_per_head(full_b, h)
        np.testing.assert_array_equal(bq, full_b[h * d:(h + 1) * d])

    def test_values_bk(self, slice_mod):
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        full_b = np.arange(3 * E, dtype=np.float64)
        h = 2
        _, bk, _ = slice_mod.slice_bias_per_head(full_b, h)
        np.testing.assert_array_equal(bk, full_b[E + h * d:E + (h + 1) * d])


class TestSliceBiasKvPerHead:
    def test_shape(self, slice_mod):
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        bk_full = np.arange(E, dtype=np.float32)
        bv_full = np.arange(E, dtype=np.float32) + E
        for h in range(slice_mod.N_HEADS):
            bk, bv = slice_mod.slice_bias_kv_per_head(bk_full, bv_full, h)
            assert bk.shape == (d,)
            assert bv.shape == (d,)

    def test_values(self, slice_mod):
        E = slice_mod.E_DIM
        d = slice_mod.D_HEAD
        bk_full = np.arange(E, dtype=np.float64) * 0.1
        bv_full = np.arange(E, dtype=np.float64) * 0.2
        h = 1
        bk, bv = slice_mod.slice_bias_kv_per_head(bk_full, bv_full, h)
        np.testing.assert_array_equal(bk, bk_full[h * d:(h + 1) * d])
        np.testing.assert_array_equal(bv, bv_full[h * d:(h + 1) * d])
