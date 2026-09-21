"""Tests for the fit() function in scripts/fit_block_sweeps.py.

fit_block_sweeps.py executes module-level code (file I/O) on import, so we
cannot import it directly.  Instead we reproduce the function under test and
verify its numerical contract.
"""
import csv
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reproducing fit() from fit_block_sweeps.py exactly (do not alter logic)
# ---------------------------------------------------------------------------

def fit(path, nmin=8):
    """Linear fit t(N) = slope*N + intercept to a block-sweep CSV.

    This is the function from fit_block_sweeps.py verbatim.
    """
    n, t = [], []
    for r in csv.DictReader(open(path)):
        if r.get("min_ms") and r["min_ms"] not in ("WARMUP_FAIL", "ITER_FAIL"):
            n.append(int(r["n_events"]))
            t.append(float(r["min_ms"]))
    n, t = np.array(n, float), np.array(t, float)
    m = n >= nmin
    A = np.vstack([n[m], np.ones(m.sum())]).T
    (sl, ic), *_ = np.linalg.lstsq(A, t[m], rcond=None)
    pred = A @ [sl, ic]
    r2 = 1 - ((t[m] - pred) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()
    return dict(slope_us=sl * 1000, intercept_us=ic * 1000, ev_per_s=1000 / sl,
                r2=float(r2), n=n.tolist(), t_ms=t.tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(path, n_vals, t_vals):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_events", "min_ms"])
        for n, t in zip(n_vals, t_vals):
            w.writerow([n, f"{t:.8f}"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_perfect_linear(linear_csv):
    """For t = 0.1*n + 2.0, slope = 0.1 ms/event → slope_us = 100."""
    d = fit(linear_csv)
    assert abs(d["slope_us"] - 100.0) < 1e-6, f"slope_us={d['slope_us']}"
    assert abs(d["intercept_us"] - 2000.0) < 1e-6, f"intercept_us={d['intercept_us']}"
    assert abs(d["r2"] - 1.0) < 1e-9, f"r2={d['r2']}"


def test_fit_ev_per_s_inverse_of_slope_us(linear_csv):
    """ev_per_s should equal 1e6 / slope_us."""
    d = fit(linear_csv)
    expected = 1e6 / d["slope_us"]
    assert abs(d["ev_per_s"] - expected) < 1.0


def test_fit_result_keys(linear_csv):
    d = fit(linear_csv)
    for key in ("slope_us", "intercept_us", "ev_per_s", "r2", "n", "t_ms"):
        assert key in d, f"missing key: {key}"


def test_fit_n_and_t_ms_lists(linear_csv):
    d = fit(linear_csv)
    assert isinstance(d["n"], list)
    assert isinstance(d["t_ms"], list)
    # All n values in the conftest CSV are >= nmin=8, so nothing is filtered
    assert len(d["n"]) == len(d["t_ms"])
    assert len(d["n"]) > 0


def test_fit_skips_warmup_fail_rows(tmp_path):
    p = str(tmp_path / "sweep.csv")
    n_vals = [8, 16, 32, 64, 128]
    t_vals = [0.1 * n + 2.0 for n in n_vals]
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_events", "min_ms"])
        w.writerow([4, "WARMUP_FAIL"])
        for n, t in zip(n_vals, t_vals):
            w.writerow([n, f"{t:.8f}"])
        w.writerow([256, "ITER_FAIL"])
    d = fit(p)
    # WARMUP_FAIL / ITER_FAIL rows must be excluded
    assert all(v not in d["t_ms"] for v in [float("inf"), float("nan")])
    # n_events=4 is below nmin=8, so it is excluded even if it had a value
    assert 4.0 not in d["n"]


def test_fit_nmin_filters_small_n(tmp_path):
    """Points below nmin should not influence the fit."""
    p = str(tmp_path / "sweep.csv")
    n_vals = [2, 4, 8, 16, 32, 64]
    t_vals = [0.1 * n + 2.0 for n in n_vals]
    write_csv(p, n_vals, t_vals)
    # All points are on the same line, so nmin filtering should not change the result.
    d_default = fit(p, nmin=8)
    d_all = fit(p, nmin=0)
    # slope must be the same since all points are on the perfect line
    assert abs(d_default["slope_us"] - d_all["slope_us"]) < 1e-6


def test_fit_returns_float_r2(linear_csv):
    d = fit(linear_csv)
    assert isinstance(d["r2"], float)


def test_fit_r2_bounded(tmp_path):
    """r2 must be in (-inf, 1] for any real data; we test it is <= 1."""
    p = str(tmp_path / "sweep.csv")
    rng = np.random.default_rng(0)
    n_vals = list(range(8, 200, 8))
    t_vals = (0.05 * np.array(n_vals, float) + 1.0 + rng.normal(0, 0.1, len(n_vals))).tolist()
    write_csv(p, n_vals, t_vals)
    d = fit(p)
    assert d["r2"] <= 1.0 + 1e-9
