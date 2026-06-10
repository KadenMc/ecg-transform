from typing import Any, List, Optional
from copy import deepcopy

import numpy as np

from scipy.sparse import spdiags, eye  # Detrend
from scipy.sparse.linalg import splu  # Detrend

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
        N = signal.shape[1]

        # Second-order difference operator D, shape (N-2, N), kept SPARSE.
        data = np.array([
            np.ones(N),
            -2 * np.ones(N),
            np.ones(N),
        ])
        D = spdiags(data, [0, 1, 2], N - 2, N)

        # Smoothness-priors detrending (Tarvainen et al. 2002):
        #     z_stat = (I - (I + λ²DᵀD)^-1) y
        # Solve (I + λ²DᵀD) x = y for every lead with a sparse LU rather than
        # forming a dense N×N inverse (np.linalg.inv on an N×N matrix is ~N²
        # memory and OOMs / runs for seconds at N≈5000, e.g. a 10 s @ 500 Hz lead).
        A = (eye(N, format='csc')
             + (self.smoothing_factor ** 2) * (D.T @ D)).tocsc()
        lu = splu(A)
        yT = np.ascontiguousarray(signal.T)   # (N, num_leads)
        smoothed = lu.solve(yT)               # (N, num_leads), one solve per lead
        detrended_signal = (yT - smoothed).T  # (num_leads, N)

        return ECGInput(detrended_signal, metadata)
