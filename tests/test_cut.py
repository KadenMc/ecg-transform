"""Tests for ecg_transform.t.cut (Pad, Crop, SegmentNonoverlapping)."""
from __future__ import annotations

import numpy as np
import pytest

from ecg_transform.t.cut import Pad, Crop, SegmentNonoverlapping
from helpers import make_input, sine_12lead


def test_pad_right():
    sig = np.ones((12, 600), dtype=np.float32)
    out = Pad(pad_to_num_samples=1000, value=0.0)([make_input(sig)])[0]
    assert out.signal.shape[1] == 1000
    assert np.all(out.signal[:, 600:] == 0)
    assert out.meta.num_samples == 1000


def test_crop_right():
    sig = sine_12lead(1200)
    out = Crop(crop_to_num_samples=1000)([make_input(sig)])[0]
    assert out.signal.shape[1] == 1000
    np.testing.assert_array_equal(out.signal, sig[:, :1000])


def test_segment_nonoverlapping_offsets_are_parent_relative():
    # Bug #4: input_start must be parent_start + local start (matches Crop).
    sig = sine_12lead(2500)
    segs = SegmentNonoverlapping(segment_length=1000)(
        [make_input(sig, input_start=100)]
    )
    assert len(segs) == 2                      # remainder (500) dropped
    assert segs[0].meta.input_start == 100
    assert segs[1].meta.input_start == 1100
    assert segs[1].meta.input_end == 2100


def test_segment_nonoverlapping_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        SegmentNonoverlapping(segment_length=0)
