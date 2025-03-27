# ecg-transform

## Installation
`pip install ecg-transform`

## Example
Here is an example of defining an input schema and transforms,
```
from ecg_transform.inp import ECGInputSchema
from ecg_transform.t.common import LinearResample, ReorderLeads
from ecg_transform.t.normalize import MinMaxNormalize
from ecg_transform.t.cut import Pad, SegmentNonoverlapping

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SAMPLE_RATE = 500
N_SAMPLES = SAMPLE_RATE*10

SCHEMA = ECGInputSchema(
    sample_rate=SAMPLE_RATE,
    expected_lead_order=LEAD_ORDER,
    required_num_samples=N_SAMPLES,
)

TRANSFORMS = [
    ReorderLeads(
        expected_order=LEAD_ORDER,
        missing_lead_strategy='raise',
    ),
    LinearResample(desired_sample_rate=SAMPLE_RATE),
    MinMaxNormalize(),
    SegmentNonoverlapping(segment_length=N_SAMPLES),
    Pad(pad_to_num_samples=N_SAMPLES, value=0)
]
```

Here is an example of how `ecg-transform` could be used in PyTorch (which we do not require to minimize dependencies),
```
from typing import List
from itertools import chain

from scipy.io import loadmat

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ecg_transform.inp import ECGInput, ECGInputSchema
from ecg_transform.t.base import ECGTransform
from ecg_transform.sample import ECGMetadata, ECGSample

class ECGDataset(Dataset):
    def __init__(
        self,
        schema,
        transforms,
        file_paths,
    ):
        self.schema = schema
        self.transforms = transforms
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mat = loadmat(self.file_paths[idx])
        metadata = ECGMetadata(
            sample_rate=int(mat['org_sample_rate'][0, 0]),
            num_samples=mat['feats'].shape[1],
            lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            unit=None,
            input_start=0,
            input_end=mat['feats'].shape[1],
        )
        inp = ECGInput(mat['feats'], metadata)
        sample = ECGSample(
            inp,
            self.schema,
            self.transforms,
        )

        return torch.from_numpy(sample.out).float(), self.file_paths[idx]

def collate_fn(inps):
    sample_ids = list(
        chain.from_iterable([[inp[1]]*inp[0].shape[0] for inp in inps])
    )
    return torch.concatenate([inp[0] for inp in inps]), sample_ids

def file_paths_to_loader(
    file_paths: List[str],
    schema: ECGInputSchema,
    transforms: List[ECGTransform],
    batch_size = 64,
    num_workers = 7,
):
    dataset = ECGDataset(
        schema,
        transforms,
        file_paths,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
```