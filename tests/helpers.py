"""Shared test helpers for ecg-transform."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ecg_transform.inp import ECGInput, ECGMetadata

LEADS12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def make_input(
    signal: np.ndarray,
    *,
    sample_rate: float = 500,
    lead_names: Optional[List[str]] = None,
    unit: str = 'mV',
    input_start: int = 0,
) -> ECGInput:
    """Build an ECGInput with a consistent ECGMetadata from a [n_leads, N] array."""
    signal = np.asarray(signal)
    leads = list(lead_names) if lead_names is not None else LEADS12[:signal.shape[0]]
    meta = ECGMetadata(
        sample_rate=sample_rate,
        num_samples=signal.shape[1],
        lead_names=leads,
        unit=unit,
        input_start=input_start,
        input_end=input_start + signal.shape[1],
    )
    return ECGInput(signal, meta)


def sine_12lead(n_samples: int = 1000, dtype=np.float32) -> np.ndarray:
    """A non-degenerate 12-lead signal (each lead a phase-shifted sine)."""
    t = np.linspace(0, 1, n_samples, endpoint=False)
    return np.stack([
        np.sin(2 * np.pi * (5 + i) * t + i) for i in range(12)
    ]).astype(dtype)
