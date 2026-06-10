"""Tests for ecg_transform.t.filter (LowpassFilter guards)."""
from __future__ import annotations

import numpy as np
import pytest

from ecg_transform.t.filter import LowpassFilter
from helpers import make_input, sine_12lead


def test_lowpass_runs_and_preserves_shape():
    out = LowpassFilter(cutoff_freq=40, order=4)(
        [make_input(sine_12lead(1000), sample_rate=500)]
    )[0]
    assert out.signal.shape == (12, 1000)
    assert np.all(np.isfinite(out.signal))


def test_lowpass_rejects_cutoff_above_nyquist():
    # Bug #6: cutoff >= Nyquist previously raised a cryptic butter error.
    with pytest.raises(ValueError, match="Nyquist"):
        LowpassFilter(cutoff_freq=300, order=4)(
            [make_input(sine_12lead(1000), sample_rate=500)]
        )


def test_lowpass_rejects_too_short_signal():
    # Bug #6: filtfilt padlen crash on short signals.
    with pytest.raises(ValueError, match="too short"):
        LowpassFilter(cutoff_freq=40, order=4)(
            [make_input(sine_12lead(10), sample_rate=500)]
        )
