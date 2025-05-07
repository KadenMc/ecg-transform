from typing import Any, Callable, List, Optional
from copy import deepcopy

import numpy as np

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

class Standardize(ECGTransform):
    """
    Subtract per-lead mean and divide by per-lead standard deviation.
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
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

        meta.unit = 'standardized'
        return ECGInput(scaled, meta)

class MinMaxNormalize(ECGTransform):
    """
    Shift each lead to zero and scale to the [0, 1] range.
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
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

        meta.unit = 'min_max_normalized'
        return ECGInput(scaled, meta)

class IQRNormalize(ECGTransform):
    """
    Normalize each lead by its interquartile range (IQR).
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
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

        meta.unit = 'iqr_normalized'
        return ECGInput(scaled, meta)
