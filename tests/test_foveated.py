"""Foveated transform shape and circular field."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.foveated import OUT, foveate


def test_foveated_shape_and_mask():
    img = np.random.default_rng(0).integers(0, 10, (256, 256, 3))
    out, mask = foveate(img)
    assert out.shape == (OUT, OUT, 3)
    assert mask.shape == (OUT, OUT)
    assert mask[OUT // 2, OUT // 2]
    assert not mask[0, 0]
    assert out.min() >= 0 and out.max() <= 9


if __name__ == "__main__":
    test_foveated_shape_and_mask()
    print("ok  test_foveated_shape_and_mask")
    print("\n1 test passed")
