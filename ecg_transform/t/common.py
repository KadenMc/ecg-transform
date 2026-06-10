from typing import Any, Callable, List, Optional
from copy import deepcopy

import numpy as np

from scipy.interpolate import interp1d  # LinearResample

from ecg_transform.inp import ECGInput
from ecg_transform.t.base import ECGTransform

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

        current_leads = list(metadata.lead_names)
        # Already in the requested order — no-op.
        if current_leads == self.expected_order:
            return inp

        missing = set(self.expected_order) - set(current_leads)
        if missing and self.missing_lead_strategy == 'raise':
            raise ValueError(
                f"Missing leads: {missing}. Set `missing_lead_strategy` to 'zero' or "
                "'constant' in ReorderLeads to fill them instead of raising."
            )

        # Fill value for any expected lead absent from the input. Under 'raise'
        # there are no missing leads at this point, so `fill` is never used.
        if self.missing_lead_strategy == 'constant':
            fill = self.missing_leads_constant
        else:  # 'zero' or 'raise' (pure reorder)
            fill = 0

        # Preserve the input dtype — do NOT hardcode float64. Upcasting an int16
        # or float32 input silently breaks downstream byte-for-byte expectations
        # (e.g. an FFT resample applied to float64 vs float32 input differs).
        lead_to_idx = {lead: idx for idx, lead in enumerate(current_leads)}
        new_signal = np.full(
            (len(self.expected_order), signal.shape[1]), fill, dtype=signal.dtype,
        )
        for idx, lead in enumerate(self.expected_order):
            if lead in lead_to_idx:
                new_signal[idx] = signal[lead_to_idx[lead]]

        metadata.lead_names = list(self.expected_order)

        return ECGInput(new_signal, metadata)

class LinearResample(ECGTransform):
    """Resample to ``desired_sample_rate`` via linear interpolation, preserving
    wall-clock duration (output length scales by the rate ratio).

    ``input_start``/``input_end`` are expressed in samples at the signal's
    *current* sample rate, so they are rescaled by the same rate ratio. For a
    fixed-length / fixed-grid resample (e.g. model input pinned to N samples),
    use :class:`~ecg_transform.t.unit.FourierResampleToLength` instead — linear
    interpolation lacks anti-aliasing and diverges from an FFT resample at sharp
    QRS peaks.
    """

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
            metadata.input_start * (self.desired_sample_rate/current_fs)
        )
        metadata.input_end = int(
            metadata.input_end * (self.desired_sample_rate/current_fs)
        )

        return ECGInput(new_signal, metadata)

class MissingLeadToConstant(ECGTransform):
    def __init__(self, leads_to_set: List[str], value: Any):
        self.leads_to_set = leads_to_set
        self.value = value

    def _transform(self, inp: ECGInput) -> ECGInput:
        # Copy before writing — never mutate the caller's array in place.
        signal = inp.signal.copy()
        metadata = deepcopy(inp.meta)
        lead_to_idx = {lead: idx for idx, lead in enumerate(metadata.lead_names)}
        for lead in self.leads_to_set:
            if lead in lead_to_idx:
                signal[lead_to_idx[lead]] = self.value
        return ECGInput(signal, metadata)

class HandleConstantLeads(ECGTransform):
    """
    Zero‐out or retain any lead whose values are constant across time.

    Parameters
    ----------
    constant_lead_strategy : str, default 'zero'
        How to handle constant leads:
            - 'zero': Set all samples in constant leads to 0.
            - 'ignore': Leave signal unchanged.
            - 'nan': Set all samples in constant leads to NaN.
    """

    def __init__(self, strategy: str = 'zero'):
        if not strategy in ['zero', 'ignore', 'nan']:
            raise ValueError(f"Unknown constant lead strategy: {strategy}.")

        self.strategy = strategy

    def _transform(self, inp: ECGInput) -> ECGInput:
        """
        Inspect inp.signal (shape: n_leads × length), detect any lead
        where every sample equals the first sample (i.e. constant), and
        apply the chosen strategy.

        Returns a new ECGInput with exactly the same metadata.
        """
        if self.strategy == 'ignore':
            return inp

        # Copy before writing — never mutate the caller's array in place.
        signal = inp.signal.copy()
        meta = deepcopy(inp.meta)

        # Boolean mask, shape (n_leads,), of leads that are constant across time.
        # No `.squeeze()`: for a single-lead [1, N] signal it would collapse to a
        # 0-d array and break the boolean row-indexing below.
        constant = np.std(signal, axis=1) == 0

        if not np.any(constant):
            # If no constant leads, nothing to be done
            return ECGInput(signal, meta)

        if self.strategy == 'zero':
            # Convert constant leads to zero
            signal[constant, :] = 0
        elif self.strategy == 'nan':
            # Convert constant leads to NaN
            signal[constant, :] = np.nan

        return ECGInput(signal, meta)
