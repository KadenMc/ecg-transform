from typing import List, Optional, Any
from dataclasses import dataclass

import numpy as np

@dataclass
class ECGInputSchema:
    """Defines the preprocessing requirements for a specific model."""
    sample_rate: float  # Desired sampling rate in Hz
    expected_lead_order: Optional[List[str]] = None  # Expected order required_num_samplesof leads, if any
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

    @property
    def length_sec(self) -> float:
        """Computes the length of the signal in seconds."""
        return self.num_samples / self.sample_rate

@dataclass
class ECGInput:
    signal: np.ndarray # Shape: (num_leads, num_samples) - even for single-lead data
    meta: ECGMetadata

    def __post_init__(self):
        assert len(self.meta.lead_names) == self.signal.shape[0]
