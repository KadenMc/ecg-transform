from typing import List, Union
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from ecg_transform.inp import ECGInput, ECGInputSchema, ECGMetadata
from ecg_transform.t.common import ECGTransform
from ecg_transform.utils.common import to_list

class ECGSample:
    def __init__(
        self,
        input_org: Union[ECGInput, List[ECGInput]],
        schema: ECGInputSchema,
        transforms: List[ECGTransform],
        debug: bool = False,
    ):
        self.transforms = transforms
        self.input_org = input_org
        self.schema = schema
        self.debug = debug
        self.input_processed = self._apply_transforms()
        self._validate_processed()

    def _apply_transforms(
        self,
    ) -> List[ECGInput]:
        processed = deepcopy(to_list(self.input_org))
        for transform in self.transforms:
            processed = transform(processed)
            if self.debug:
                print(f"After transform: {transform.__class__.__name__}")
                for i, inp in enumerate(processed):
                    print(f"Input {i}:")
                    print(f"  - Lead means: {inp.signal.mean(axis=1)}.")
                    print(f"  - Metadata: {inp.meta}")

        return processed

    def _validate_processed(self):
        for inp in to_list(self.input_processed):
            # Sampling rate check
            if inp.meta.sample_rate != self.schema.sample_rate:
                raise ValueError(
                    f"Sampling rate mismatch: {inp.meta.sample_rate} != {self.schema.sample_rate}"
                )
            # Lead order check
            if self.schema.expected_lead_order is not None:
                if inp.meta.lead_names != self.schema.expected_lead_order:
                    raise ValueError(
                        f"Lead order mismatch: {inp.meta.lead_names} != {self.schema.expected_lead_order}"
                    )
            # NaN checks
            if not self.schema.allow_nan_samples and np.isnan(inp.signal).any():
                raise ValueError("NaN values in signal not allowed.")
            if not self.schema.allow_nan_leads:
                nan_leads = np.isnan(inp.signal).all(axis=1)
                if nan_leads.any():
                    raise ValueError(
                        f"Leads with all NaNs not allowed: {np.array(inp.meta.lead_names)[nan_leads]}"
                    )

    @property
    def out(self):
        return np.stack([inp.signal for inp in self.input_processed])
