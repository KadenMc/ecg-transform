from typing import Any, Callable, List, Optional
from copy import deepcopy

import numpy as np

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

class Standardize(ECGTransform):
    def __init__(self, constant_lead_strategy: str = 'zero'):
        self.constant_lead_strategy = constant_lead_strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)
        mean = np.mean(signal, axis=1, keepdims=True)
        signal = signal - mean
        std = np.std(signal, axis=1, keepdims=True)
        constant = std == 0
        if not constant.any() or self.constant_lead_strategy == 'nan':
            signal = signal / std
        else:
            std_replaced = np.where(constant, 1, std)
            signal = signal / std_replaced
            if self.constant_lead_strategy == 'zero':
                signal[constant] = 0
            elif self.constant_lead_strategy == 'keep':
                pass
            else:
                raise ValueError(
                    f"Unknown constant_lead_strategy: {self.constant_lead_strategy}"
                )

        metadata.unit = 'standardized'

        return ECGInput(signal, metadata)

class MinMaxNormalize(ECGTransform):
    def __init__(self, constant_lead_strategy: str = 'zero'):
        self.constant_lead_strategy = constant_lead_strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)

        signal_min = np.min(signal, axis=1, keepdims=True)
        signal_max = np.max(signal, axis=1, keepdims=True)
        constant = (signal_min == signal_max).squeeze()

        if not constant.any() or self.constant_lead_strategy == 'nan':
            signal = (signal - signal_min)/(signal_max - signal_min)
        else:
            signal = (signal - signal_min)/(signal_max - signal_min + 1e-8)
            if self.constant_lead_strategy == 'zero':
                signal[constant] = 0
            elif self.constant_lead_strategy == 'keep':
                pass
            else:
                raise ValueError(
                    f"Unknown constant_lead_strategy: {self.constant_lead_strategy}"
                )

        metadata.unit = 'min_max_normalized'

        return ECGInput(signal, metadata)

class IQRNormalize(ECGTransform):
    """
    A transform class that normalizes ECG signals using the Interquartile Range (IQR).
    This is typically used as a form of outlier removal.

    Args:
        constant_lead_strategy (str): Strategy for handling leads with zero IQR.
            Options are 'zero' (set to zero) or 'keep' (retain normalized values).
            Default is 'zero'.
    """
    def __init__(self, constant_lead_strategy: str = 'zero'):
        self.constant_lead_strategy = constant_lead_strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        """
        Applies IQR normalization to the ECG signal.
        
        Args:
            inp (ECGInput): Input ECG object with signal and metadata.
        
        Returns:
            ECGInput: New ECGInput object with normalized signal and updated metadata.
        """
        # Extract the signal (shape: num_leads, signal_length)
        signal = inp.signal

        # Create a deep copy of metadata to avoid modifying the original
        metadata = deepcopy(inp.meta)

        # Compute 25th and 75th percentiles for each lead
        Q1 = np.percentile(signal, 25, axis=1, keepdims=True)
        Q3 = np.percentile(signal, 75, axis=1, keepdims=True)

        # Calculate IQR per lead
        IQR = Q3 - Q1

        # Small constant to prevent division by zero
        epsilon = 1e-8

        # Normalize the signal: (signal - Q1) / (IQR + epsilon)
        normalized_signal = (signal - Q1) / (IQR + epsilon)

        # Handle constant leads (where IQR == 0)
        if self.constant_lead_strategy == 'zero':
            constant = (IQR == 0).squeeze()
            if constant.any():
                # Set normalized values to zero for constant leads
                normalized_signal[constant] = 0
        elif self.constant_lead_strategy != 'keep':
            raise ValueError(f"Unknown constant_lead_strategy: {self.constant_lead_strategy}")
        # If 'keep', retain the normalized values (which are zero for constant leads)

        # Update metadata to indicate the signal is IQR normalized
        metadata.unit = 'iqr_normalized'

        # Return new ECGInput object with transformed signal
        return ECGInput(normalized_signal, metadata)
