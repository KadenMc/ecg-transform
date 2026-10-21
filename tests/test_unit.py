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


# The micro sign (U+00B5) and the Greek small mu (U+03BC): both spell the prefix.
MICRO_SIGN, GREEK_MU = chr(0x00B5), chr(0x03BC)


@pytest.mark.parametrize(
    'unit',
    ['uV', 'UV', MICRO_SIGN + 'V', GREEK_MU + 'V', 'microvolt', 'microvolts', 'Microvolts'],
)
def test_convert_unit_microvolts_to_mv(unit):
    sig = np.full((12, 8), 1500.0, dtype=np.float32)  # microvolts
    out = ConvertUnit('mV')([make_input(sig, unit=unit)])[0]
    np.testing.assert_allclose(out.signal, 1.5)
    assert out.meta.unit == 'mV'
    assert out.signal.dtype == np.float32


@pytest.mark.parametrize('unit', ['V', 'v', 'volt', 'volts', 'Volts'])
def test_convert_unit_volts_to_mv(unit):
    sig = np.full((12, 8), 0.0015, dtype=np.float32)  # volts
    out = ConvertUnit('mV')([make_input(sig, unit=unit)])[0]
    np.testing.assert_allclose(out.signal, 1.5, rtol=1e-6)
    assert out.meta.unit == 'mV'
    assert out.signal.dtype == np.float32


def test_convert_unit_microvolts_divide_with_one_rounding():
    # Dividing by 1000 rounds once, so values a float32 can hold in mV come out exact.
    sig = np.array([[1500.0, -250.0, 0.0, 125.0]], dtype=np.float32)
    out = ConvertUnit('mV')([make_input(sig, unit='uV')])[0]
    assert np.array_equal(out.signal, np.array([[1.5, -0.25, 0.0, 0.125]], dtype=np.float32))


def test_convert_unit_physical_units_ignore_a_gain():
    # A gain belongs to counts; microvolts and volts are fixed units and need none.
    sig = np.full((12, 8), 1000.0, dtype=np.float32)
    out = ConvertUnit('mV', adc_microvolts_per_count=4.88)(
        [make_input(sig, unit='uV', adc_microvolts_per_count=4.88)]
    )[0]
    np.testing.assert_allclose(out.signal, 1.0)


def test_convert_unit_physical_units_keep_the_rest_and_leave_the_input_alone():
    inp = make_input(sine_12lead(10), unit='uV', sample_rate=250, input_start=3)
    out = ConvertUnit('mV')([inp])[0]
    assert out is not inp
    assert (out.meta.sample_rate, out.meta.num_samples, out.meta.input_start, out.meta.input_end) == (250, 10, 3, 13)
    assert out.meta.lead_names == inp.meta.lead_names
    assert inp.meta.unit == 'uV'
    np.testing.assert_allclose(out.signal, inp.signal / 1000.0)


def test_convert_unit_still_refuses_other_units():
    for bad in ('mmV', 'nV', 'kV', 'uV/count', 'degreeC'):
        with pytest.raises(ValueError, match="'uV'"):
            ConvertUnit('mV')([make_input(sine_12lead(8), unit=bad)])


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
