"""1960x1080 tiled geometry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.tiled_1960 import CH, CW, TH, TW, S, make_canvas


def test_tile_geometry():
    assert TH * 10 == CH
    assert TW * 10 == CW
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 10, (S, S, 3))
    canvas = make_canvas([patch], 0, 3, 4, rng)
    assert canvas.shape == (CH, CW, 3)
    x = 3 * TW + (TW - S) // 2
    y = 4 * TH + (TH - S) // 2
    assert np.array_equal(canvas[y:y + S, x:x + S], patch)


if __name__ == "__main__":
    test_tile_geometry()
    print("ok  test_tile_geometry")
    print("\n1 test passed")
