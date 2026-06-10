from typing import Any, List, Optional
from copy import deepcopy

import numpy as np

from scipy.signal import butter, filtfilt  # LowpassFilter

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

class LowpassFilter(ECGTransform):
    def __init__(self, cutoff_freq: float, order: int = 4):
        self.cutoff_freq = cutoff_freq
        self.order = order

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)
        fs = metadata.sample_rate
        nyquist = 0.5 * fs
        normal_cutoff = self.cutoff_freq / nyquist
        if not 0 < normal_cutoff < 1:
            raise ValueError(
                f"cutoff_freq ({self.cutoff_freq} Hz) must be in the open interval "
                f"(0, Nyquist={nyquist} Hz) for a sample_rate of {fs} Hz."
            )
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        # filtfilt needs the signal longer than its default padlen, else it raises
        # a cryptic error. Guard explicitly (common after segmentation).
        padlen = 3 * max(len(a), len(b))
        if signal.shape[1] <= padlen:
            raise ValueError(
                f"Signal length ({signal.shape[1]}) is too short for filtfilt with "
                f"order={self.order} (needs > {padlen} samples). Lower the order or "
                "filter before segmenting."
            )
        filtered_signal = np.array([filtfilt(b, a, lead) for lead in signal])
        return ECGInput(filtered_signal, metadata)