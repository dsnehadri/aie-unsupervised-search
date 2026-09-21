"""Shared pytest fixtures for the aie-unsupervised-search test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def perfect_sep_scores():
    """Perfectly separated: all signal scores > all background scores."""
    bkg = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    sig = np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    return bkg, sig


@pytest.fixture
def identical_scores():
    """Identical score distributions (each array same values)."""
    bkg = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    sig = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    return bkg, sig


@pytest.fixture
def reversed_scores():
    """All background scores above all signal scores — worst-case discriminant."""
    bkg = np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    sig = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    return bkg, sig


@pytest.fixture
def random_scores(rng):
    bkg = rng.uniform(0, 1, 300)
    sig = rng.uniform(0.2, 1.2, 300)
    return bkg, sig


@pytest.fixture
def linear_csv(tmp_path):
    """A CSV that obeys t = 0.1*n + 2.0 exactly (ms).
    Used to validate fit() in fit_block_sweeps.py."""
    p = tmp_path / "sweep.csv"
    lines = ["n_events,min_ms"]
    for n in [8, 16, 32, 64, 128, 256, 512]:
        t = 0.1 * n + 2.0
        lines.append(f"{n},{t:.6f}")
    p.write_text("\n".join(lines))
    return str(p)
