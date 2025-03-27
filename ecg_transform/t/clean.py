from typing import Any, List, Optional
from copy import deepcopy

import numpy as np

from scipy.sparse import spdiags # Detrend

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

class Detrend(ECGTransform):
    """A transformation class to detrend ECG signals using a second-order difference method.

    Args:
        smoothing_factor (float): Smoothing parameter controlling the detrending strength.
    """
    def __init__(self, smoothing_factor: float):
        self.smoothing_factor = smoothing_factor

    def _transform(self, inp: ECGInput) -> ECGInput:
        """Apply detrending to the ECG signal.

        Args:
            inp (ECGInput): Input ECG data with a 2D signal array and metadata.

        Returns:
            ECGInput: New ECGInput object with the detrended signal and updated metadata.
        """
        signal = inp.signal  # Shape: (num_leads, signal_length)
        metadata = deepcopy(inp.meta)
        signal_length = signal.shape[1]

        # Identity matrix
        H = np.identity(signal_length)

        # Second-order difference matrix
        data = np.array([
            np.ones(signal_length),
            -2 * np.ones(signal_length),
            np.ones(signal_length)
        ])
        D = spdiags(data, [0, 1, 2], signal_length - 2, signal_length).toarray()

        # Transformation matrix
        M = H - np.linalg.inv(H + (self.smoothing_factor ** 2) * (D.T @ D))

        # Detrend all leads
        detrended_signal = (M @ signal.T).T  # Shape: (num_leads, signal_length)

        return ECGInput(detrended_signal, metadata)
