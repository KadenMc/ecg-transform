from copy import deepcopy

import numpy as np

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

_VALID_CONSTANT_LEAD_STRATEGIES = ('zero', 'nan', 'ignore')


def _apply_constant_lead_strategy(
    scaled: np.ndarray, original: np.ndarray, strategy: str,
) -> np.ndarray:
    """Post-normalization handling of leads that were constant in ``original``.

    A constant lead has zero range/variance, so every normalizer maps it to a
    constant (≈0) lead via the ``+ 1e-8`` denominator guard. This makes that
    handling explicit and configurable:

    - ``'zero'``  : set constant leads to 0 (the implicit default behaviour).
    - ``'nan'``   : set constant leads to NaN (flag them as non-informative).
    - ``'ignore'``: leave the normalized values untouched.
    """
    if strategy == 'ignore':
        return scaled
    constant = np.std(original, axis=1) == 0
    if not np.any(constant):
        return scaled
    scaled = scaled.copy()
    if strategy == 'zero':
        scaled[constant, :] = 0
    elif strategy == 'nan':
        scaled[constant, :] = np.nan
    return scaled


class Standardize(ECGTransform):
    """
    Subtract per-lead mean and divide by per-lead standard deviation.
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
        if constant_lead_strategy not in _VALID_CONSTANT_LEAD_STRATEGIES:
            raise ValueError(
                f"Unknown constant_lead_strategy: {constant_lead_strategy!r} "
                f"(expected one of {_VALID_CONSTANT_LEAD_STRATEGIES})."
            )
        self.constant_lead_strategy = constant_lead_strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        """
        Apply zero-mean and unit-variance scaling, then handle any constant leads.
        """
        signal: np.ndarray = inp.signal
        meta = deepcopy(inp.meta)

        # subtract per-lead mean
        mean = np.mean(signal, axis=1, keepdims=True)
        centered = signal - mean

        # divide by per-lead std, adding a small epsilon to avoid division by zero
        std = np.std(centered, axis=1, keepdims=True)
        scaled = centered / (std + 1e-8)

        scaled = _apply_constant_lead_strategy(scaled, signal, self.constant_lead_strategy)
        meta.unit = 'standardized'
        return ECGInput(scaled, meta)

class MinMaxNormalize(ECGTransform):
    """
    Shift each lead to zero and scale to the [0, 1] range.
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
        if constant_lead_strategy not in _VALID_CONSTANT_LEAD_STRATEGIES:
            raise ValueError(
                f"Unknown constant_lead_strategy: {constant_lead_strategy!r} "
                f"(expected one of {_VALID_CONSTANT_LEAD_STRATEGIES})."
            )
        self.constant_lead_strategy = constant_lead_strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        """
        Apply min-max normalization, then handle any constant leads.
        """
        signal: np.ndarray = inp.signal
        meta = deepcopy(inp.meta)

        # compute per-lead min and max
        mn = np.min(signal, axis=1, keepdims=True)
        mx = np.max(signal, axis=1, keepdims=True)

        # shift and scale, adding epsilon to avoid division by zero
        scaled = (signal - mn) / ((mx - mn) + 1e-8)

        scaled = _apply_constant_lead_strategy(scaled, signal, self.constant_lead_strategy)
        meta.unit = 'min_max_normalized'
        return ECGInput(scaled, meta)

class IQRNormalize(ECGTransform):
    """
    Normalize each lead by its interquartile range (IQR).
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
        if constant_lead_strategy not in _VALID_CONSTANT_LEAD_STRATEGIES:
            raise ValueError(
                f"Unknown constant_lead_strategy: {constant_lead_strategy!r} "
                f"(expected one of {_VALID_CONSTANT_LEAD_STRATEGIES})."
            )
        self.constant_lead_strategy = constant_lead_strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        """
        Apply IQR normalization, then handle any constant leads.
        """
        signal: np.ndarray = inp.signal
        meta = deepcopy(inp.meta)

        # compute first and third quartiles
        Q1 = np.percentile(signal, 25, axis=1, keepdims=True)
        Q3 = np.percentile(signal, 75, axis=1, keepdims=True)
        IQR = Q3 - Q1

        # shift by Q1 and scale by IQR, adding epsilon to avoid division by zero
        scaled = (signal - Q1) / (IQR + 1e-8)

        scaled = _apply_constant_lead_strategy(scaled, signal, self.constant_lead_strategy)
        meta.unit = 'iqr_normalized'
        return ECGInput(scaled, meta)
