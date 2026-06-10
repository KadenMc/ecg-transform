"""Tests for ecg_transform.t.common (regression repros for PR1 bug fixes)."""
from __future__ import annotations

import numpy as np
import pytest

from ecg_transform.t.common import (
    ReorderLeads,
    LinearResample,
    MissingLeadToConstant,
    HandleConstantLeads,
)
from helpers import make_input, LEADS12, sine_12lead


# --- ReorderLeads -----------------------------------------------------------

def test_reorder_reorders_values():
    shuffled = LEADS12[::-1]
    sig = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    out = ReorderLeads(LEADS12, 'raise')([make_input(sig, lead_names=shuffled)])[0]
    assert out.meta.lead_names == LEADS12
    # lead 'I' is last in the shuffled input -> row 11
    np.testing.assert_array_equal(out.signal[0], sig[11])


def test_reorder_raise_on_missing_lead():
    # Bug #1b: 'raise' + missing lead must raise (previously fell through to garbage).
    leads = LEADS12[:-1]  # drop V6
    sig = sine_12lead(8)[:11]
    with pytest.raises(ValueError, match="Missing leads"):
        ReorderLeads(LEADS12, 'raise')([make_input(sig, lead_names=leads)])


def test_reorder_zero_fills_missing():
    leads = LEADS12[:-1]
    sig = np.ones((11, 6), dtype=np.float32)
    out = ReorderLeads(LEADS12, 'zero')([make_input(sig, lead_names=leads)])[0]
    assert np.all(out.signal[-1] == 0)        # V6 filled with zeros
    assert np.all(out.signal[0] == 1)         # present lead preserved


def test_reorder_constant_fills_missing():
    # Bug #1a: 'constant' previously raised AttributeError (missing_leads_to_value).
    leads = LEADS12[:-1]
    sig = np.ones((11, 6), dtype=np.float32)
    out = ReorderLeads(LEADS12, 'constant', missing_leads_constant=7.0)(
        [make_input(sig, lead_names=leads)]
    )[0]
    assert np.all(out.signal[-1] == 7.0)


def test_reorder_constant_requires_value():
    with pytest.raises(ValueError):
        ReorderLeads(LEADS12, 'constant')  # missing_leads_constant=None


def test_reorder_preserves_dtype():
    # Bug #1d: dtype must follow the input, not hardcoded float64 (byte-identity).
    shuffled = LEADS12[::-1]
    sig = sine_12lead(8, dtype=np.float32)
    out = ReorderLeads(LEADS12, 'raise')([make_input(sig, lead_names=shuffled)])[0]
    assert out.signal.dtype == np.float32


# --- LinearResample ---------------------------------------------------------

def test_linear_resample_changes_length_and_rate():
    sig = sine_12lead(1000)
    out = LinearResample(desired_sample_rate=250)(
        [make_input(sig, sample_rate=500)]
    )[0]
    assert out.signal.shape[1] == 500
    assert out.meta.sample_rate == 250
    assert out.meta.num_samples == 500


# --- MissingLeadToConstant --------------------------------------------------

def test_missing_lead_to_constant_does_not_mutate_input():
    # Bug #3: must not mutate the caller's array in place.
    sig = np.ones((12, 5), dtype=np.float32)
    inp = make_input(sig)
    out = MissingLeadToConstant(['II'], value=9.0)([inp])[0]
    assert np.all(out.signal[1] == 9.0)       # II set on output
    assert np.all(inp.signal == 1)            # original untouched


# --- HandleConstantLeads ----------------------------------------------------

def test_handle_constant_leads_nan_strategy():
    # Bug #2: 'nan' strategy was dead (branch checked 'keep').
    sig = sine_12lead(20)
    sig[3] = 0.0  # make lead 3 constant
    out = HandleConstantLeads('nan')([make_input(sig)])[0]
    assert np.all(np.isnan(out.signal[3]))
    assert not np.any(np.isnan(out.signal[0]))


def test_handle_constant_leads_single_lead():
    # Bug #2: .squeeze() broke single-lead [1, N].
    sig = np.zeros((1, 16), dtype=np.float32)  # one constant lead
    out = HandleConstantLeads('zero')([make_input(sig, lead_names=['II'])])[0]
    assert out.signal.shape == (1, 16)
    assert np.all(out.signal == 0)


def test_handle_constant_leads_does_not_mutate_input():
    sig = sine_12lead(16)
    sig[2] = 0.0
    inp = make_input(sig)
    HandleConstantLeads('nan')([inp])
    assert not np.any(np.isnan(inp.signal))   # original untouched
