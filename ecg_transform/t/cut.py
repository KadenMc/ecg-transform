from typing import Any, Callable, List, Optional
from copy import deepcopy

import numpy as np

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

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

class SegmentNonoverlapping(ECGTransform):
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
            segment_metadata.input_start = start
            segment_metadata.input_end = end

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
        segments = []
        for boundary in boundaries:
            # Define segment start and end, ensuring they stay within signal bounds
            start = max(0, boundary - self.left_offset)
            end = min(num_samples, boundary + self.right_offset)

            # Only include non-empty segments
            if start < end:
                segment_signal = signal[:, start:end]
                segment_metadata = deepcopy(metadata)
                segment_metadata.num_samples = end - start
                segment_metadata.input_start = start
                segment_metadata.input_end = end
                segments.append(ECGInput(segment_signal, segment_metadata))

        return segments
