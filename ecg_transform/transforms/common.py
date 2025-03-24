from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union
from copy import deepcopy

import numpy as np

from scipy.interpolate import interp1d  # LinearResample
from scipy.signal import butter, filtfilt  # LowpassFilter

from ecg_transform.inp import ECGInput
from ecg_transform.transforms.base import ECGTransform

class ReorderLeads(ECGTransform):
    def __init__(
        self,
        expected_order: List[str],
        missing_lead_strategy: str = 'raise',
        missing_leads_constant: Optional[Any] = None,
    ):
        self.expected_order = expected_order
        self.missing_lead_strategy = missing_lead_strategy
        self.missing_leads_constant = missing_leads_constant

        assert missing_lead_strategy in ['raise', 'zero', 'constant']
        if missing_lead_strategy == 'constant' and missing_leads_constant is None:
            raise ValueError(
                "Must specify `missing_leads_constant` when using strategy 'constant'."
            )

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)

        current_leads = metadata.lead_names
        if set(current_leads) == set(self.expected_order) and list(current_leads) == self.expected_order:
            return inp

        if not self.missing_lead_strategy == 'raise' and \
            not set(current_leads).issuperset(self.expected_order):
            missing = set(self.expected_order) - set(current_leads)
            raise ValueError(
                f"Missing leads: {missing}. Can change `missing_lead_strategy` in ReorderLeads transform."
            )

        lead_to_idx = {lead: idx for idx, lead in enumerate(current_leads)}
        if self.missing_lead_strategy == 'raise':
            new_signal = np.empty((len(self.expected_order), signal.shape[1]), dtype='float64')
        else:
            new_signal = np.full(
                (len(self.expected_order), signal.shape[1]),
                0 if self.missing_lead_strategy == 'zero' else self.missing_leads_to_value,
                dtype='float64',
            )

        for idx, lead in enumerate(self.expected_order):
            if lead in lead_to_idx:
                new_signal[idx] = signal[lead_to_idx[lead]]

        metadata.lead_names = self.expected_order

        return ECGInput(new_signal, metadata)

class Pad(ECGTransform):
    def __init__(self, pad_to_num_samples: int, value: float, direction: str = 'right'):
        """
        Initialize the Pad transformation.

        Args:
            pad_to_num_samples (int): Desired total number of samples after padding.
            value (float): Constant value to use for padding the signal.
            direction (str, optional): Direction to apply padding ('left' or 'right'). Defaults to 'right'.
        """
        # Validate direction
        assert direction in ['left', 'right'], "Direction must be 'left' or 'right'"
        # Validate pad_to_num_samples
        assert isinstance(pad_to_num_samples, int) and pad_to_num_samples >= 0, \
            "pad_to_num_samples must be a non-negative integer"
        
        self.pad_to_num_samples = pad_to_num_samples
        self.value = value
        self.direction = direction

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)
        current_num_samples = signal.shape[1]

        # Check if padding is needed
        if current_num_samples >= self.pad_to_num_samples:
            return inp  # No padding needed if signal is already long enough

        # Calculate padding amount
        pad_amount = self.pad_to_num_samples - current_num_samples

        # Determine padding configuration based on direction
        if self.direction == 'left':
            pad_config = ((0, 0), (pad_amount, 0))  # Pad left side
            metadata.input_start -= pad_amount
        elif self.direction == 'right':
            pad_config = ((0, 0), (0, pad_amount))  # Pad right side
            metadata.input_end += pad_amount

        # Apply padding to the signal
        new_signal = np.pad(
            signal,
            pad_width=pad_config,
            mode='constant',
            constant_values=self.value
        )

        # Update metadata
        metadata.num_samples = self.pad_to_num_samples

        # Return new ECGInput with padded signal and updated metadata
        return ECGInput(new_signal, metadata)

class Crop(ECGTransform):
    def __init__(self, crop_to_num_samples: int, direction: str = 'right'):
        # Validate direction
        assert direction in ['left', 'right'], "Direction must be 'left' or 'right'"
        # Validate crop_to_num_samples
        assert isinstance(crop_to_num_samples, int) and crop_to_num_samples >= 0, \
            "crop_to_num_samples must be a non-negative integer"

        self.crop_to_num_samples = crop_to_num_samples
        self.direction = direction

    def _transform(self, inp: ECGInput) -> ECGInput:
        # Extract signal and create a deep copy of metadata
        signal = inp.signal  # Shape: (num_leads, num_samples)
        metadata = deepcopy(inp.meta)
        current_num_samples = signal.shape[1]

        # Check if cropping is needed
        if current_num_samples <= self.crop_to_num_samples:
            return inp  # No cropping needed if signal is already short enough

        # Calculate number of samples to keep
        crop_amount = current_num_samples - self.crop_to_num_samples

        # Determine cropping indices based on direction
        if self.direction == 'left':
            # Keep samples from the right end
            new_signal = signal[:, crop_amount:]
            metadata.input_start += crop_amount
        elif self.direction == 'right':
            # Keep samples from the left end
            new_signal = signal[:, :-crop_amount]
            metadata.input_end -= crop_amount

        # Update metadata
        metadata.num_samples = self.crop_to_num_samples

        # Return new ECGInput with cropped signal and updated metadata
        return ECGInput(new_signal, metadata)

class LinearResample(ECGTransform):
    def __init__(self, desired_sample_rate: float):
        self.desired_sample_rate = desired_sample_rate

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)
        current_fs = metadata.sample_rate
        if current_fs == self.desired_sample_rate:
            return inp

        num_samples = signal.shape[1] if signal.ndim > 1 else signal.shape[0]
        desired_num_samples = int(
            num_samples * (self.desired_sample_rate / current_fs)
        )
        x = np.linspace(0, desired_num_samples - 1, num_samples)
        interp_func = interp1d(x, signal, kind='linear', axis=-1)
        new_signal = interp_func(np.arange(desired_num_samples))
        metadata.sample_rate = self.desired_sample_rate
        metadata.num_samples = desired_num_samples
        metadata.input_start = int(
            metadata.input_start * (desired_sample_rate/current_fs)
        )
        metadata.input_end = int(
            metadata.input_end * (desired_sample_rate/current_fs)
        )

        return ECGInput(new_signal, metadata)

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

class MissingLeadToConstant(ECGTransform):
    def __init__(self, leads_to_set: List[str], value: Any):
        self.leads_to_set = leads_to_set
        self.value = value

    def _transform(self, inp: ECGInput) -> ECGInput:
        signal = inp.signal
        metadata = deepcopy(inp.meta)
        lead_to_idx = {lead: idx for idx, lead in enumerate(metadata.lead_names)}
        for lead in self.leads_to_set:
            if lead in lead_to_idx:
                signal[lead_to_idx[lead]] = self.value
        return ECGInput(signal, metadata)

class LowpassFilter(ECGTransform):
    def __init__(self, cutoff_freq: float, order: int = 5):
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

class Segment(ECGTransform):
    def __init__(self, segment_length: int):
        self.segment_length = segment_length

    def _transform(self, inp: ECGInput) -> List[ECGInput]:
        signal = inp.signal
        metadata = deepcopy(inp.meta)
        num_samples = signal.shape[1]
        num_segments = num_samples // self.segment_length
        segments = []
        for i in range(num_segments):
            start = i * self.segment_length
            end = start + self.segment_length
            segment_signal = signal[:, start:end]
            segment_metadata = deepcopy(metadata)
            segment_metadata.num_samples = self.segment_length
            segments.append(ECGInput(segment_signal, segment_metadata))

        return segments

class SegmentOnBoundaries(ECGTransform):
    """
    Segments an ECG signal around dynamically computed boundary points with optional offsets.

    Args:
        boundary_fn (Callable[[ECGInput], List[int]]): Function accepting ECGInput and returning boundary samples.
        left_offset (int, optional): Number of samples to extend left of each boundary. Defaults to 0.
        right_offset (int, optional): Number of samples to extend right of each boundary. Defaults to 0.
    """
    def __init__(
        self,
        boundary_fn: Callable[[ECGInput], List[int]],
        left_offset: int = 0,
        right_offset: int = 0,
    ):
        self.boundary_fn = boundary_fn
        self.left_offset = left_offset
        self.right_offset = right_offset

    def _transform(self, inp: ECGInput) -> List[ECGInput]:
        # Compute boundaries dynamically using the provided function
        boundaries = self.boundary_fn(inp)

        signal = inp.signal  # Shape: (num_leads, num_samples)
        metadata = deepcopy(inp.meta)
        num_samples = signal.shape[1]

        for boundary in boundaries:
            # Define segment start and end, ensuring they stay within signal bounds
            start = max(0, boundary - self.left_offset)
            end = min(num_samples, boundary + self.right_offset)

            # Only include non-empty segments
            if start < end:
                segment_signal = signal[:, start:end]
                segment_metadata = deepcopy(metadata)
                segment_metadata.num_samples = end - start
                segments.append(ECGInput(segment_signal, segment_metadata))

        return segments
