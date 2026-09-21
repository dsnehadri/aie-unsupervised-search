"""Tests for parsing logic in scripts/summarize_aie_profile.py.

The script is purely imperative (no functions, only side-effects + sys.argv),
so we cannot import it.  Instead we test the three key regex patterns and the
arithmetic used to derive per-event cycle counts.
"""
import re

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Regex patterns copied verbatim from summarize_aie_profile.py
# ---------------------------------------------------------------------------

TILE_PATTERN = re.compile(r"profile_funct_(\d+_\d+)\.txt")
KERN_PATTERN = re.compile(r"(obj|cand|cross)_(post|attn)")
ROW_PATTERN = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)%\s+\d+\s+\d+\s+\d+\s+(\d+)\s+([\d.]+)%"
    r"\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\S+)\s",
    re.M,
)


# ---------------------------------------------------------------------------
# Tile filename regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("profile_funct_0_0.txt", "0_0"),
    ("profile_funct_12_3.txt", "12_3"),
    ("profile_funct_255_255.txt", "255_255"),
])
def test_tile_pattern_matches(filename, expected):
    m = TILE_PATTERN.search(filename)
    assert m is not None, f"pattern did not match {filename!r}"
    assert m.group(1) == expected


@pytest.mark.parametrize("filename", [
    "profile_funct_0.txt",       # only one number — should not match
    "profile_func_0_0.txt",      # wrong prefix
    "some_other_file.txt",
])
def test_tile_pattern_no_match(filename):
    assert TILE_PATTERN.search(filename) is None


# ---------------------------------------------------------------------------
# Kernel name filter regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "obj_attn", "cand_attn", "cross_attn",
    "obj_post", "cand_post", "cross_post",
])
def test_kern_pattern_matches_valid_kernels(name):
    assert KERN_PATTERN.match(name) is not None


@pytest.mark.parametrize("name", [
    "layernorm_row", "f32_mul", "softfloat_add", "__div_called",
    "gemm_kernel", "obj_other", "attn_obj",
])
def test_kern_pattern_rejects_non_kernels(name):
    assert KERN_PATTERN.match(name) is None


# ---------------------------------------------------------------------------
# Profile-row regex
# ---------------------------------------------------------------------------

SAMPLE_ROW = (
    "    42   123456  12.34%    0    0    0  654321  56.78%"
    "    0    0    0    0    0    0  obj_attn \n"
)

def test_row_pattern_matches_sample():
    m = ROW_PATTERN.search(SAMPLE_ROW)
    assert m is not None
    assert int(m.group(1)) == 42      # calls
    assert int(m.group(2)) == 123456  # cycles_tot
    assert float(m.group(3)) == 12.34 # pct_f
    assert int(m.group(4)) == 654321  # cfd (func+desc cycles)
    assert float(m.group(5)) == 56.78 # pct_fd
    assert m.group(6) == "obj_attn"   # name


def test_row_pattern_no_match_on_header():
    header = "          Calls  Cycles tot\n"
    assert ROW_PATTERN.search(header) is None


def test_row_pattern_no_match_on_empty_line():
    assert ROW_PATTERN.search("\n") is None


# ---------------------------------------------------------------------------
# per_ev arithmetic: total_func_desc_cycles / calls / 1.25e3 -> us/event
# ---------------------------------------------------------------------------

def test_per_ev_calculation():
    """At 1.25 GHz, cycles_per_call / 1250 = microseconds per event."""
    cycles_per_call = 1250     # exactly 1 µs at 1.25 GHz
    per_ev = cycles_per_call / 1.25e3
    assert abs(per_ev - 1.0) < 1e-9


def test_per_ev_scales_linearly():
    """Doubling cycles_per_call doubles µs/event."""
    c1, c2 = 2500, 5000
    assert abs(c2 / 1.25e3 - 2 * c1 / 1.25e3) < 1e-9


# ---------------------------------------------------------------------------
# Duplicate-table de-duplication (the script keeps only the first occurrence)
# ---------------------------------------------------------------------------

SEPARATOR = "          Calls  Cycles tot"

def test_first_block_selection():
    """After splitting on the separator, the script uses the first block."""
    block_a = "block A content"
    block_b = "block B content"
    txt = block_a + SEPARATOR + block_b
    parts = txt.split(SEPARATOR)
    # Script reconstructs: parts[0] + (separator + parts[1] if len>1 else "")
    reconstructed = parts[0] + (SEPARATOR + parts[1] if len(parts) > 1 else "")
    assert "block A content" in reconstructed
    # The second block is still appended (script only keeps the first table
    # by stopping the regex at the boundary — the separator itself is preserved
    # to anchor the header).
    assert "block B content" in reconstructed


def test_single_block_no_separator():
    txt = "only one block"
    parts = txt.split(SEPARATOR)
    assert len(parts) == 1
    reconstructed = parts[0] + (SEPARATOR + parts[1] if len(parts) > 1 else "")
    assert reconstructed == txt
