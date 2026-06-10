"""Tests for ecg_transform.t.unit (ConvertUnit, FourierResampleToLength)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import resample

from ecg_transform.t.unit import ConvertUnit, FourierResampleToLength
from helpers import make_input, sine_12lead


# --- ConvertUnit ------------------------------------------------------------

def test_convert_unit_mv_is_noop():
    inp = make_input(sine_12lead(100), unit='mV')
    out = ConvertUnit('mV')([inp])[0]
    assert out is inp
    assert out.meta.unit == 'mV'


def test_convert_unit_adc_to_mv_corpus_constant():
    # Corpus-wide constant passed to the transform.
    sig = np.full((12, 8), 1000.0, dtype=np.float32)  # ADC counts
    out = ConvertUnit('mV', adc_microvolts_per_count=4.88)(
        [make_input(sig, unit='adc')]
    )[0]
    np.testing.assert_allclose(out.signal, 1000.0 * (4.88 / 1000.0))
    assert out.meta.unit == 'mV'


def test_convert_unit_adc_gain_from_metadata():
    # Per-record gain on ECGMetadata, no constant on the transform.
    sig = np.full((12, 8), 1000.0, dtype=np.float32)
    out = ConvertUnit('mV')(
        [make_input(sig, unit='adc', adc_microvolts_per_count=4.88)]
    )[0]
    np.testing.assert_allclose(out.signal, 1000.0 * (4.88 / 1000.0))
    assert out.meta.unit == 'mV'


def test_convert_unit_constant_overrides_metadata():
    sig = np.full((12, 8), 1000.0, dtype=np.float32)
    out = ConvertUnit('mV', adc_microvolts_per_count=4.88)(
        [make_input(sig, unit='adc', adc_microvolts_per_count=1.0)]
    )[0]
    np.testing.assert_allclose(out.signal, 1000.0 * (4.88 / 1000.0))  # 4.88, not 1.0


def test_convert_unit_adc_requires_gain():
    # ADC counts with neither a constant nor per-record metadata gain -> raise.
    with pytest.raises(ValueError, match="gain"):
        ConvertUnit('mV')([make_input(sine_12lead(8), unit='adc')])


def test_convert_unit_adc_preserves_float32():
    sig = sine_12lead(8, dtype=np.float32)
    out = ConvertUnit('mV', adc_microvolts_per_count=4.88)(
        [make_input(sig, unit='adc')]
    )[0]
    assert out.signal.dtype == np.float32


def test_convert_unit_raises_on_unknown_unit():
    for bad in (None, 'standardized', 'foo'):
        with pytest.raises(ValueError):
            ConvertUnit('mV')([make_input(sine_12lead(8), unit=bad)])


def test_convert_unit_rejects_non_mv_target():
    with pytest.raises(ValueError):
        ConvertUnit('uV')


# --- FourierResampleToLength ------------------------------------------------

def test_fourier_resample_changes_length():
    out = FourierResampleToLength(1000)([make_input(sine_12lead(5000))])[0]
    assert out.signal.shape == (12, 1000)
    assert out.meta.num_samples == 1000


def test_fourier_resample_preserves_dtype():
    out = FourierResampleToLength(1000)(
        [make_input(sine_12lead(5000, dtype=np.float32))]
    )[0]
    assert out.signal.dtype == np.float32


def test_fourier_resample_matches_scipy_byte_identical():
    sig = sine_12lead(5000, dtype=np.float32)
    out = FourierResampleToLength(1000)([make_input(sig)])[0]
    ref = resample(sig, 1000, axis=1).astype(np.float32)
    assert np.array_equal(out.signal, ref)


def test_fourier_resample_noop_when_already_length():
    inp = make_input(sine_12lead(1000))
    out = FourierResampleToLength(1000)([inp])[0]
    assert out is inp  # no FFT round-trip


def test_fourier_resample_stamps_sample_rate():
    # 10.41 s @ 500 Hz -> 5206 samples; resample to 1000, declare target 100 Hz.
    out = FourierResampleToLength(1000, sample_rate=100)(
        [make_input(sine_12lead(5206), sample_rate=500)]
    )[0]
    assert out.meta.sample_rate == 100
    assert out.signal.shape[1] == 1000


def test_fourier_resample_rejects_nonpositive():
    with pytest.raises(ValueError):
        FourierResampleToLength(0)
