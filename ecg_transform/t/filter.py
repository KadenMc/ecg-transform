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
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_signal = np.array([filtfilt(b, a, lead) for lead in signal])
        return ECGInput(filtered_signal, metadata)



def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y