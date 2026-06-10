"""Tests for ecg_transform.t.clean (Detrend sparse refactor)."""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import spdiags

from ecg_transform.t.clean import Detrend
from helpers import make_input


def _dense_detrend(signal: np.ndarray, lam: float) -> np.ndarray:
    """The original dense reference implementation, for equivalence checking."""
    N = signal.shape[1]
    H = np.identity(N)
    data = np.array([np.ones(N), -2 * np.ones(N), np.ones(N)])
    D = spdiags(data, [0, 1, 2], N - 2, N).toarray()
    M = H - np.linalg.inv(H + (lam ** 2) * (D.T @ D))
    return (M @ signal.T).T


def test_detrend_matches_dense_reference_small_n():
    rng = np.random.default_rng(0)
    sig = (np.linspace(0, 5, 150)[None, :] + 0.1 * rng.standard_normal((2, 150)))
    out = Detrend(smoothing_factor=10.0)([make_input(sig)])[0]
    ref = _dense_detrend(sig, 10.0)
    np.testing.assert_allclose(out.signal, ref, atol=1e-6)


def test_detrend_removes_linear_trend():
    n = 300
    trend = np.linspace(0, 6, n)  # asymmetric ramp -> clearly non-zero mean (~3)
    sig = (trend + np.sin(np.linspace(0, 20 * np.pi, n)))[None, :]
    out = Detrend(smoothing_factor=50.0)([make_input(sig)])[0]
    # detrended signal should have near-zero mean (the ramp is gone)
    assert abs(out.signal.mean()) < 0.1
    assert abs(out.signal.mean()) < abs(sig.mean())
    assert out.signal.shape == sig.shape


def test_detrend_scales_to_large_n():
    # Bug #8: dense N×N inverse OOMs/stalls at N=5000; sparse solve must be quick.
    sig = np.random.default_rng(1).standard_normal((12, 5000))
    t0 = time.time()
    out = Detrend(smoothing_factor=300.0)([make_input(sig)])[0]
    assert out.signal.shape == (12, 5000)
    assert np.all(np.isfinite(out.signal))
    assert time.time() - t0 < 20  # generous; dense inv would be far worse
