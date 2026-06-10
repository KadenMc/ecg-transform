"""Tests for ecg_transform.t.scale (normalizers + constant_lead_strategy)."""
from __future__ import annotations

import numpy as np
import pytest

from ecg_transform.t.scale import Standardize, MinMaxNormalize, IQRNormalize
from helpers import make_input, sine_12lead


def test_standardize_zero_mean_unit_std():
    out = Standardize()([make_input(sine_12lead(500))])[0]
    np.testing.assert_allclose(out.signal.mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(out.signal.std(axis=1), 1, atol=1e-3)
    assert out.meta.unit == 'standardized'


def test_minmax_in_unit_range():
    out = MinMaxNormalize()([make_input(sine_12lead(500))])[0]
    assert out.signal.min() >= 0.0 - 1e-6
    assert out.signal.max() <= 1.0 + 1e-6
    assert out.meta.unit == 'min_max_normalized'


def test_iqr_runs():
    out = IQRNormalize()([make_input(sine_12lead(500))])[0]
    assert out.signal.shape == (12, 500)
    assert out.meta.unit == 'iqr_normalized'


def test_constant_lead_strategy_nan():
    # Bug #9: the param was accepted but never applied.
    sig = sine_12lead(64)
    sig[5] = 0.0  # constant lead
    out = MinMaxNormalize(constant_lead_strategy='nan')([make_input(sig)])[0]
    assert np.all(np.isnan(out.signal[5]))
    assert not np.any(np.isnan(out.signal[0]))


def test_constant_lead_strategy_zero():
    sig = sine_12lead(64)
    sig[5] = 3.0  # constant lead
    out = Standardize(constant_lead_strategy='zero')([make_input(sig)])[0]
    assert np.all(out.signal[5] == 0)


def test_constant_lead_strategy_invalid():
    with pytest.raises(ValueError):
        Standardize(constant_lead_strategy='bogus')
