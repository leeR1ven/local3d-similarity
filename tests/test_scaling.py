"""Regression test for the 0..9 levels -> network input conversion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.scaling_clean import to_levels


def test_levels_are_not_divided_by_255():
    images = np.array([[[0, 5, 9], [1, 6, 8]]], dtype=np.int64)
    levels = to_levels(images)
    assert levels.dtype.kind in "iu"
    assert levels.max() == 9
    assert levels.min() == 0
    assert not np.all(levels == 0)


if __name__ == "__main__":
    test_levels_are_not_divided_by_255()
    print("ok  test_levels_are_not_divided_by_255")
    print("\n1 test passed")
