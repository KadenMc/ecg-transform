from typing import List, Optional, Any
from dataclasses import dataclass

import numpy as np

@dataclass
class ECGInputSchema:
    """Defines the preprocessing requirements for a specific model."""
    sample_rate: float  # Desired sampling rate in Hz
    expected_lead_order: Optional[List[str]] = None  # Expected order of leads, if any
    partial_leads: bool = False  # Whether missing leads are allowed
    missing_leads_to_value: Optional[Any] = None  # Value to fill missing leads, if allowed
    allow_nan_samples: bool = False  # Whether NaN values are allowed in the signal
    allow_nan_leads: bool = False  # Whether entirely NaN leads are allowed
    required_num_samples: Optional[int] = None  # Exact number of samples, if required
    min_num_samples: Optional[int] = None  # Minimum acceptable number of samples
    max_num_samples: Optional[int] = None  # Maximum acceptable number of samples

@dataclass
class ECGMetadata:
    sample_rate: float
    num_samples: int
    lead_names: List[str]
    unit: str
    input_start: int # Sample # of the start of the input relative to the original sample
    input_end: int # Sample # of the end of the input relative to the original sample
    # ADC gain in microvolts per count, for ADC→mV conversion (see
    # ecg_transform.t.unit.ConvertUnit). Optional and per-record: leave None for
    # signals already in physical units, or when a single corpus-wide constant is
    # supplied to ConvertUnit instead. This is the right home for real,
    # sample-specific gains (e.g. read from a WFDB header).
    adc_microvolts_per_count: Optional[float] = None

    @property
    def length_sec(self) -> float:
        """Computes the length of the signal in seconds."""
        return self.num_samples / self.sample_rate

@dataclass
class ECGInput:
    signal: np.ndarray # Shape: (num_leads, num_samples) - even for single-lead data
    meta: ECGMetadata

    def __post_init__(self):
        if self.signal.ndim != 2:
            raise ValueError(
                f"ECG signal must be 2-D [n_leads, n_samples], got ndim={self.signal.ndim} "
                f"(shape={self.signal.shape})."
            )
        if len(self.meta.lead_names) != self.signal.shape[0]:
            raise ValueError(
                f"lead_names length ({len(self.meta.lead_names)}) does not match the "
                f"number of signal leads ({self.signal.shape[0]})."
            )
        if self.meta.num_samples != self.signal.shape[1]:
            raise ValueError(
                f"meta.num_samples ({self.meta.num_samples}) does not match the signal "
                f"sample count ({self.signal.shape[1]})."
            )
