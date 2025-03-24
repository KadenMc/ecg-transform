from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union
from copy import deepcopy

import numpy as np

from scipy.interpolate import interp1d  # LinearResample
from scipy.signal import butter, filtfilt  # LowpassFilter

from ecg_transform.inp import ECGInput

class ECGTransform(ABC):
    def __call__(self, ecg_data_list: List[ECGInput]) -> List[ECGInput]:
        """
        Apply the transform to each ECGInput in the list.
        Handles both single-output and multi-output transforms by flattening the results.
        """
        result = []
        for ecg_data in ecg_data_list:
            transformed = self._transform(ecg_data)
            if isinstance(transformed, list):
                result.extend(transformed)
            else:
                result.append(transformed)
        return result