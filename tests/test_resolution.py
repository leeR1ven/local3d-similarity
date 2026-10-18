"""Resolution rendering keeps the fixed model input shape."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.resolution_test import OUT, render


def test_render_shape_and_range():
    src = np.random.default_rng(0).integers(0, 10, (256, 256, 3))
    for res in (48, 128, 256):
        out = render(src, res)
        assert out.shape == (OUT, OUT, 3)
        assert out.min() >= 0 and out.max() <= 9


if __name__ == "__main__":
    test_render_shape_and_range()
    print("ok  test_render_shape_and_range")
    print("\n1 test passed")
