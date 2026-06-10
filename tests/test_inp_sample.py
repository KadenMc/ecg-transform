"""Tests for ECGInput validation and ECGSample.out guards."""
from __future__ import annotations

import numpy as np
import pytest

from ecg_transform.inp import ECGInput, ECGInputSchema, ECGMetadata
from ecg_transform.sample import ECGSample
from helpers import make_input, LEADS12, sine_12lead


# --- ECGInput.__post_init__ -------------------------------------------------

def test_ecginput_rejects_1d_signal():
    meta = ECGMetadata(500, 10, ['II'], 'mV', 0, 10)
    with pytest.raises(ValueError):
        ECGInput(np.zeros(10), meta)


def test_ecginput_rejects_lead_count_mismatch():
    meta = ECGMetadata(500, 10, ['I', 'II'], 'mV', 0, 10)
    with pytest.raises(ValueError):
        ECGInput(np.zeros((3, 10)), meta)


def test_ecginput_rejects_num_samples_mismatch():
    meta = ECGMetadata(500, 999, ['II'], 'mV', 0, 999)
    with pytest.raises(ValueError):
        ECGInput(np.zeros((1, 10)), meta)


# --- ECGSample.out ----------------------------------------------------------

def _schema(n):
    return ECGInputSchema(
        sample_rate=500, expected_lead_order=LEADS12, required_num_samples=n,
    )


def test_ecgsample_out_normal():
    inp = make_input(sine_12lead(1000))
    sample = ECGSample(inp, _schema(1000), [])
    assert sample.out.shape == (1, 12, 1000)


def test_ecgsample_out_empty_raises():
    # SegmentNonoverlapping with segment longer than signal -> empty list.
    from ecg_transform.t.cut import SegmentNonoverlapping
    inp = make_input(sine_12lead(500))
    sample = ECGSample(inp, _schema(2000), [SegmentNonoverlapping(2000)])
    with pytest.raises(ValueError, match="empty"):
        _ = sample.out


def test_ecgsample_out_ragged_raises():
    a = make_input(sine_12lead(1000))
    b = make_input(sine_12lead(800))
    # schema only checks sample_rate/lead_order/NaN, not length, so validation
    # passes and the ragged stack is caught by .out.
    schema = ECGInputSchema(sample_rate=500, expected_lead_order=LEADS12)
    sample = ECGSample([a, b], schema, [])
    with pytest.raises(ValueError, match="differing shapes"):
        _ = sample.out


def test_ecgsample_validates_sample_rate():
    inp = make_input(sine_12lead(1000), sample_rate=250)
    with pytest.raises(ValueError, match="Sampling rate"):
        ECGSample(inp, _schema(1000), [])
