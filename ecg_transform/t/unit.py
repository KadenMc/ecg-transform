"""Unit-aware amplitude conversion and fixed-length Fourier resampling.

These are kept separate from ``ecg_transform.t.scale`` (statistical
normalization) because they preserve / target *physical* units and a fixed
sample grid, which is what a model whose front-end is amplitude-sensitive
(e.g. with frozen BatchNorm statistics calibrated in mV) requires.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform


class ConvertUnit(ECGTransform):
    """Convert a signal to a target physical unit, driven by ``meta.unit``.

    The source declares its unit on ``ECGMetadata.unit``; this transform brings
    it to ``target``. Making the conversion contingent on the declared unit is
    the whole point — it makes "ADC counts treated as mV" (or vice versa)
    structurally impossible rather than a silent scale error.

    The ADC gain is **not** a property of this transform's configuration; it is a
    property of the data. There is no device-specific default. Supply it one of
    two ways when converting ADC counts → mV:

    - **Per-record (real, sample-specific):** set
      ``ECGMetadata.adc_microvolts_per_count`` on each input (e.g. read from a
      WFDB header). Construct ``ConvertUnit('mV')`` with no gain.
    - **Corpus-wide constant (mock / no per-record metadata):** pass
      ``adc_microvolts_per_count=<µV/count>`` here; it applies to every input.

    If a constant is passed it takes precedence over per-record metadata. If a
    signal is in ADC counts and neither is available, a ``ValueError`` is raised.

    Parameters
    ----------
    target : str, default 'mV'
        Target unit. Only ``'mV'`` is currently supported.
    adc_microvolts_per_count : float, optional
        Corpus-wide ADC gain in microvolts per count. ``None`` (default) defers
        to each record's ``ECGMetadata.adc_microvolts_per_count``. (For GE MUSE
        the value is 4.88 µV/count — but that belongs to the caller's data, not
        to this library.)

    Behaviour by ``meta.unit`` (case-insensitive):
    - ``'mV'``                  -> no-op (returns the input unchanged)
    - ``'adc'`` / ``'counts'``  -> multiply by ``gain / 1000``, set
      ``meta.unit = 'mV'`` (gain = constant if given, else per-record metadata)
    - anything else (incl. ``None``) -> raise ``ValueError``
    """

    _MV_ALIASES = frozenset({'mv'})
    _ADC_ALIASES = frozenset({'adc', 'counts'})

    def __init__(self, target: str = 'mV',
                 adc_microvolts_per_count: Optional[float] = None):
        if target != 'mV':
            raise ValueError(f"ConvertUnit only supports target='mV', got {target!r}.")
        self.target = target
        self.adc_microvolts_per_count = adc_microvolts_per_count

    def _transform(self, inp: ECGInput) -> ECGInput:
        unit = (inp.meta.unit or '').lower()
        if unit in self._MV_ALIASES:
            return inp  # already mV — no-op, signal untouched
        if unit in self._ADC_ALIASES:
            # Constant (corpus-wide) wins; otherwise per-record metadata.
            gain = self.adc_microvolts_per_count
            if gain is None:
                gain = inp.meta.adc_microvolts_per_count
            if gain is None:
                raise ValueError(
                    "ConvertUnit: converting from ADC counts needs a gain. Either "
                    "pass adc_microvolts_per_count=<µV/count> (one constant for the "
                    "whole corpus) or set ECGMetadata.adc_microvolts_per_count per "
                    "record."
                )
            meta = deepcopy(inp.meta)
            # float32 * python float stays float32 (no upcast) -> byte-identity.
            signal = inp.signal * (gain / 1000.0)
            meta.unit = 'mV'
            return ECGInput(signal, meta)
        raise ValueError(
            f"ConvertUnit cannot convert from unit {inp.meta.unit!r}; expected one "
            "of 'mV', 'adc', 'counts'. Declare the source unit on ECGMetadata.unit."
        )


class FourierResampleToLength(ECGTransform):
    """Resample to an exact sample count via ``scipy.signal.resample`` (FFT).

    Unlike :class:`~ecg_transform.t.common.LinearResample` (rate-ratio, linear
    interpolation, no anti-aliasing), this pins the output to ``num_samples`` on
    a fixed grid using the Fourier method — the same method ``scipy.signal``
    provides and the right choice when downstream scores must match an FFT-based
    training/inference pipeline. Linear vs FFT resampling diverge substantially
    at sharp QRS peaks.

    Parameters
    ----------
    num_samples : int
        Exact output length.
    sample_rate : float, optional
        If given, stamp ``meta.sample_rate`` with this value (the nominal target
        grid rate). Use this when length-normalizing variable-duration inputs to
        a fixed grid: a 10.41 s recording resampled to 1000 samples is treated as
        the target rate, not ``1000 / 10.41``. If ``None``, the new rate is
        computed from the length ratio.
    """

    def __init__(self, num_samples: int, sample_rate: Optional[float] = None):
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}.")
        self.num_samples = num_samples
        self.sample_rate = sample_rate

    def _transform(self, inp: ECGInput) -> ECGInput:
        from scipy.signal import resample

        n_in = inp.signal.shape[1]
        meta = deepcopy(inp.meta)

        if n_in == self.num_samples:
            # No resample needed (avoids a needless FFT round-trip that would
            # perturb an already-correct-length signal). Stamp the rate if the
            # caller pinned one and it differs.
            if self.sample_rate is None or meta.sample_rate == self.sample_rate:
                return inp
            meta.sample_rate = self.sample_rate
            return ECGInput(inp.signal, meta)

        resampled = resample(inp.signal, self.num_samples, axis=1).astype(inp.signal.dtype)
        if self.sample_rate is not None:
            meta.sample_rate = self.sample_rate
        elif n_in:
            meta.sample_rate = inp.meta.sample_rate * self.num_samples / n_in
        meta.num_samples = self.num_samples
        meta.input_end = meta.input_start + self.num_samples
        return ECGInput(resampled, meta)
