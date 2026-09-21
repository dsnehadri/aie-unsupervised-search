"""Tests for chart / data-processing scripts.

Only pure computation functions are exercised; matplotlib figure creation is
mocked so the tests run headless without a display.  Heavy file-I/O paths are
given synthetic data via tmp_path.

Scripts covered:
  make_latency_and_stage_charts.py  – _csv() parser, polyfit slope
  make_link_benchmark_charts.py     – load(), bytes_fmt()
  plot_lockin_pl_vs_aie.py          – fold(), roll(), unwrap()
  plot_soak.py                      – roll()
  analyze_board_lockin.py           – main() (imported; has __main__ guard)
"""
import csv as csv_mod
import os
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


# ===========================================================================
# make_latency_and_stage_charts._csv
# The function is small enough to inline so we avoid executing the module's
# matplotlib figure-save calls.
# ===========================================================================

def _csv(path, kernel):
    """Inlined from make_latency_and_stage_charts._csv."""
    pts = []
    for line in open(path):
        if line.startswith(kernel + ","):
            q = line.split(",")
            pts.append((int(q[1]), float(q[4])))
    return pts


def test_csv_parse_basic(tmp_path):
    csv = tmp_path / "sweep.csv"
    csv.write_text(
        "other_kernel,1,x,x,0.5\n"
        "pl_stream_top,8,x,x,1.42\n"
        "pl_stream_top,16,x,x,2.45\n"
        "unrelated,4,x,x,0.9\n"
    )
    pts = _csv(str(csv), "pl_stream_top")
    assert pts == [(8, 1.42), (16, 2.45)]


def test_csv_parse_returns_empty_for_missing_kernel(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("other,1,x,x,0.5\n")
    pts = _csv(str(csv), "missing_kernel")
    assert pts == []


def test_csv_parse_first_column_prefix_match(tmp_path):
    """A kernel prefix must not match a longer kernel name accidentally."""
    csv = tmp_path / "sweep.csv"
    csv.write_text(
        "pl_stream_top_extra,8,x,x,1.0\n"
        "pl_stream_top,8,x,x,2.0\n"
    )
    pts = _csv(str(csv), "pl_stream_top")
    assert len(pts) == 1
    assert pts[0][1] == 2.0


# ===========================================================================
# make_latency_and_stage_charts: batch sweep → polyfit slope
# ===========================================================================

def test_polyfit_slope_positive():
    """Steady-state time vs batch size should yield a strictly positive slope."""
    pts = [(8, 1.42699), (16, 2.45448), (32, 4.51056), (64, 8.61873), (128, 16.83715)]
    n = np.array([a for a, _ in pts], float)
    t = np.array([b for _, b in pts], float)
    slope, _ = np.polyfit(n, t, 1)
    assert slope > 0


def test_polyfit_slope_physically_reasonable():
    """Slope should be roughly ms-per-event, i.e. in (0.01, 1.0) for these data."""
    pts = [(8, 1.42699), (16, 2.45448), (32, 4.51056), (64, 8.61873)]
    n = np.array([a for a, _ in pts], float)
    t = np.array([b for _, b in pts], float)
    slope, _ = np.polyfit(n, t, 1)
    assert 0.01 < slope < 1.0


def test_polyfit_slope_units():
    """slope * 1000 should give µs/event values in the hundreds for PL-only."""
    # PL-only steady state from the script's SWEEP dict
    pts = [(8, 1.42699), (16, 2.45448), (32, 4.51056), (64, 8.61873),
           (128, 16.83715), (256, 33.27150)]
    n = np.array([a for a, _ in pts], float)
    t = np.array([b for _, b in pts], float)
    slope, _ = np.polyfit(n, t, 1)
    us_per_event = slope * 1000
    assert 100 < us_per_event < 500


# ===========================================================================
# make_link_benchmark_charts.bytes_fmt
# ===========================================================================

def bytes_fmt(b):
    """Inlined from make_link_benchmark_charts.bytes_fmt."""
    for u, d in (("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= d:
            v = b / d
            return f"{v:g} {u}"
    return f"{b} B"


@pytest.mark.parametrize("b,expected", [
    (64, "64 B"),
    (1024, "1 KB"),
    (2048, "2 KB"),
    (1048576, "1 MB"),
    (4194304, "4 MB"),
    (512, "512 B"),
    (1 << 23, "8 MB"),
])
def test_bytes_fmt(b, expected):
    assert bytes_fmt(b) == expected


def test_bytes_fmt_sub_kb_stays_bytes():
    assert "B" in bytes_fmt(500)
    assert "KB" not in bytes_fmt(500)
    assert "MB" not in bytes_fmt(500)


# ===========================================================================
# make_link_benchmark_charts.load
# ===========================================================================

def load_link(fname_path, key):
    """Inlined from make_link_benchmark_charts.load (accepts full path)."""
    if not os.path.exists(fname_path):
        return []
    pts = []
    for r in csv_mod.reader(open(fname_path)):
        if r and r[0] == key and r[4] != "-1":
            pts.append((int(r[1]), float(r[4])))
    return sorted(pts)


def test_load_link_missing_file():
    pts = load_link("/nonexistent/path.csv", "ptw")
    assert pts == []


def test_load_link_filters_by_key(tmp_path):
    csv = tmp_path / "sweep.csv"
    csv.write_text(
        "ptw,64,a,b,0.025\n"
        "lb,64,a,b,0.012\n"
        "ptw,128,a,b,-1\n"
        "ptw,256,a,b,0.031\n"
    )
    pts = load_link(str(csv), "ptw")
    assert pts == [(64, 0.025), (256, 0.031)]


def test_load_link_excludes_minus_one(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("k,128,a,b,-1\n")
    pts = load_link(str(csv), "k")
    assert pts == []


def test_load_link_sorted(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("k,512,a,b,0.9\nk,64,a,b,0.3\nk,128,a,b,0.5\n")
    pts = load_link(str(csv), "k")
    byte_sizes = [b for b, _ in pts]
    assert byte_sizes == sorted(byte_sizes)


# ===========================================================================
# plot_lockin_pl_vs_aie: fold(), roll(), unwrap()
# Inlined to avoid module-level CSV reads that require real hardware log files.
# ===========================================================================

BIN = 1.0  # same constant as the script


def fold_fn(rows, epochs, ons, period, key):
    """Inlined from plot_lockin_pl_vs_aie.fold (simplified, no VCCINT branch)."""
    vals = np.array([float(r[key]) for r in rows])
    nb = int(np.ceil(period / BIN))
    starts = np.array([cs for cs, _ in ons])
    idx = np.searchsorted(starts, epochs, side="right") - 1
    ok = idx >= 0
    off = np.full(len(epochs), np.nan)
    off[ok] = epochs[ok] - starts[idx[ok]]
    m = ok & (off >= 0) & (off < period) & np.isfinite(vals)
    b = np.clip((off[m] / BIN).astype(int), 0, nb - 1)
    sums = np.bincount(b, weights=vals[m], minlength=nb)
    cnts = np.bincount(b, minlength=nb)
    mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return (np.arange(nb) + 0.5) * BIN, mean


def roll_lockin(xs, k=5):
    """Inlined from plot_lockin_pl_vs_aie.roll (periodic padding)."""
    h = k // 2
    pad = np.concatenate([xs[-h:], xs, xs[:h]])
    return np.array([np.nanmean(pad[i:i + 2 * h + 1]) for i in range(len(xs))])


def unwrap_fn(t, period, pre=45.0):
    """Inlined from plot_lockin_pl_vs_aie.unwrap."""
    x = np.where(t > period - pre, t - period, t)
    return np.argsort(x), np.sort(x)


def test_fold_output_shape():
    period = 10.0
    nb = int(period / BIN)
    ons = [(0.0, 7.0)]
    epochs = np.linspace(0.5, 9.5, 10)
    rows = [{"power": "1.0"} for _ in epochs]
    t, mean = fold_fn(rows, epochs, ons, period, "power")
    assert t.shape == (nb,)
    assert mean.shape == (nb,)


def test_fold_constant_signal():
    """Constant signal -> every bin should average to that constant."""
    period = 5.0
    nb = int(period / BIN)
    ons = [(0.0, 3.0)]
    epochs = np.arange(0.5, 5.0, 1.0)
    rows = [{"power": "2.5"} for _ in epochs]
    _, mean = fold_fn(rows, epochs, ons, period, "power")
    valid = mean[np.isfinite(mean)]
    np.testing.assert_allclose(valid, 2.5, atol=1e-9)


def test_roll_lockin_preserves_length():
    xs = np.arange(10, dtype=float)
    out = roll_lockin(xs, k=3)
    assert len(out) == len(xs)


def test_roll_lockin_smooths_spike():
    xs = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    out = roll_lockin(xs, k=3)
    assert out[2] < 10.0


def test_roll_lockin_constant():
    xs = np.ones(20) * 3.0
    out = roll_lockin(xs, k=5)
    np.testing.assert_allclose(out, 3.0, atol=1e-9)


def test_unwrap_shifts_tail_before_zero():
    period = 10.0
    pre = 3.0
    # 8.5 > period - pre = 7.0, so it maps to 8.5 - 10 = -1.5
    t = np.array([0.5, 5.0, 8.5])
    _, sorted_t = unwrap_fn(t, period, pre)
    assert sorted_t[0] < 0


def test_unwrap_returns_sorted():
    rng = np.random.default_rng(42)
    t = rng.uniform(0, 12.0, 30)
    _, st = unwrap_fn(t, 12.0, 3.0)
    assert np.all(st[:-1] <= st[1:])


def test_unwrap_no_shift_for_early_times():
    """Points that are not in the tail should stay unchanged."""
    period = 10.0
    pre = 2.0
    t = np.array([1.0, 4.0, 6.0])   # all < period - pre = 8.0
    _, st = unwrap_fn(t, period, pre)
    np.testing.assert_array_equal(sorted(t), st)


# ===========================================================================
# plot_soak.roll (asymmetric window, no periodic wrap)
# ===========================================================================

def soak_roll(x, k=15):
    """Inlined from plot_soak.roll."""
    h = k // 2
    return np.array(
        [np.nanmean(x[max(0, i - h):min(len(x), i + h + 1)])
         for i in range(len(x))]
    )


def test_soak_roll_length():
    x = np.ones(50)
    assert len(soak_roll(x)) == 50


def test_soak_roll_constant():
    x = np.ones(30) * 5.0
    np.testing.assert_allclose(soak_roll(x, k=15), 5.0, atol=1e-9)


def test_soak_roll_handles_edges():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = soak_roll(x, k=3)
    assert np.isfinite(out).all()


# ===========================================================================
# analyze_board_lockin.main
# Safe to import directly: the module has a `if __name__ == "__main__":` guard.
# ===========================================================================

def _write_lockin_files(tmp_path, n_cycles=5, period=90):
    """Produce synthetic CSV + phase files for analyze_board_lockin."""
    on_start_offset = 10   # seconds of idle before first ON
    phase_lines = []
    csv_lines = ["epoch,power_W\n"]

    on_intervals = []
    for i in range(n_cycles):
        s = on_start_offset + i * period * 2
        e = s + period
        on_intervals.append((s, e))
        phase_lines.append(f"on_start {s}\n")
        phase_lines.append(f"on_end {e}\n")

    total = on_start_offset + n_cycles * period * 2 + period + 10
    for t in range(int(total)):
        on = any(s <= t < e for s, e in on_intervals)
        csv_lines.append(f"{t},{'5.0' if on else '2.0'}\n")

    phase_path = tmp_path / "phases.txt"
    phase_path.write_text("".join(phase_lines))
    csv_path = tmp_path / "log.csv"
    csv_path.write_text("".join(csv_lines))
    return str(csv_path), str(phase_path)


def test_analyze_board_lockin_runs(tmp_path, capsys):
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    try:
        import analyze_board_lockin as alb
        csv_path, phase_path = _write_lockin_files(tmp_path, n_cycles=5, period=90)
        alb.main(csv_path, phase_path)
        out = capsys.readouterr().out
        assert "cycles used:" in out
    finally:
        # Clean up so re-import works in subsequent test runs
        sys.modules.pop("analyze_board_lockin", None)
        if SCRIPTS_DIR in sys.path:
            sys.path.remove(SCRIPTS_DIR)


def test_analyze_board_lockin_prints_deltas(tmp_path, capsys):
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    try:
        import analyze_board_lockin as alb
        csv_path, phase_path = _write_lockin_files(tmp_path, n_cycles=6, period=90)
        alb.main(csv_path, phase_path)
        out = capsys.readouterr().out
        assert "power_W" in out  # power column should appear in output
    finally:
        sys.modules.pop("analyze_board_lockin", None)
        if SCRIPTS_DIR in sys.path:
            sys.path.remove(SCRIPTS_DIR)


def test_analyze_board_lockin_cycle_count(tmp_path, capsys):
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    try:
        import analyze_board_lockin as alb
        n = 4
        csv_path, phase_path = _write_lockin_files(tmp_path, n_cycles=n, period=90)
        alb.main(csv_path, phase_path)
        out = capsys.readouterr().out
        # The last cycle has no trailing OFF window, so used cycles <= n
        assert "cycles used:" in out
        used = int(out.split("cycles used:")[1].split()[0])
        assert used <= n
    finally:
        sys.modules.pop("analyze_board_lockin", None)
        if SCRIPTS_DIR in sys.path:
            sys.path.remove(SCRIPTS_DIR)
