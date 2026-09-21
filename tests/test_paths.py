"""Tests for scripts/paths.py.

paths.py only imports os and computes string constants, so it is safe
to import directly without triggering any file I/O.
"""
import importlib
import importlib.util
import os
import sys

import pytest

# Make scripts/ importable.
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import paths  # noqa: E402  (import after sys.path manipulation)


@pytest.fixture(autouse=True)
def _restore_paths():
    """Save and restore paths constants so env-override tests don't contaminate others."""
    saved_data = paths.DATA
    saved_model_repo = paths.MODEL_REPO
    yield
    paths.DATA = saved_data
    paths.MODEL_REPO = saved_model_repo


# ---------------------------------------------------------------------------
# Basic type / format checks
# ---------------------------------------------------------------------------

def test_repo_is_string():
    assert isinstance(paths.REPO, str)


def test_data_is_string():
    assert isinstance(paths.DATA, str)


def test_figs_is_string():
    assert isinstance(paths.FIGS, str)


def test_model_repo_is_string():
    assert isinstance(paths.MODEL_REPO, str)


# ---------------------------------------------------------------------------
# Structural sanity: paths should not be empty or just "/"
# ---------------------------------------------------------------------------

def test_repo_not_empty():
    assert len(paths.REPO) > 1


def test_data_not_empty():
    assert len(paths.DATA) > 1


def test_figs_not_empty():
    assert len(paths.FIGS) > 1


# ---------------------------------------------------------------------------
# REPO should point at the repository root (contains scripts/ and figs/)
# ---------------------------------------------------------------------------

def test_repo_contains_scripts_dir():
    assert os.path.isdir(os.path.join(paths.REPO, "scripts"))


def test_repo_contains_figs_dir():
    assert os.path.isdir(os.path.join(paths.REPO, "figs"))


# ---------------------------------------------------------------------------
# FIGS should be derived from REPO
# ---------------------------------------------------------------------------

def test_figs_under_repo():
    assert paths.FIGS.startswith(paths.REPO)


# ---------------------------------------------------------------------------
# Environment-variable overrides
# ---------------------------------------------------------------------------

def test_passwd_data_env_override(monkeypatch, tmp_path):
    """When PASSWD_DATA is set, DATA should equal that value."""
    monkeypatch.setenv("PASSWD_DATA", str(tmp_path))
    # importlib.reload fails on Python 3.13 for sys.path-injected modules;
    # exec_module re-runs the source in-place without needing a findable spec.
    _spec = importlib.util.spec_from_file_location("paths", paths.__file__)
    _spec.loader.exec_module(paths)
    assert paths.DATA == str(tmp_path)


def test_passwd_model_repo_env_override(monkeypatch, tmp_path):
    """When PASSWD_MODEL_REPO is set, MODEL_REPO should equal that value."""
    monkeypatch.setenv("PASSWD_MODEL_REPO", str(tmp_path))
    _spec = importlib.util.spec_from_file_location("paths", paths.__file__)
    _spec.loader.exec_module(paths)
    assert paths.MODEL_REPO == str(tmp_path)


def test_default_data_is_data_subdir():
    """Without env override, DATA should be REPO/data."""
    if "PASSWD_DATA" in os.environ:
        pytest.skip("PASSWD_DATA is set; cannot test default")
    assert paths.DATA == os.path.join(paths.REPO, "data")
