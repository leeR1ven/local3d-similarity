"""LFW fovea preserves geometry and keeps the face center sharp."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.lfw_face_foveated import H, W, foveate_face


def test_face_fovea_shape_and_center():
    img = np.random.default_rng(0).integers(0, 10, (H, W, 3))
    out = foveate_face(img)
    assert out.shape == (H, W, 3)
    assert out.min() >= 0 and out.max() <= 9
    cy, cx = H // 2, W // 2
    assert np.array_equal(out[cy - 3:cy + 3, cx - 3:cx + 3], img[cy - 3:cy + 3, cx - 3:cx + 3])


if __name__ == "__main__":
    test_face_fovea_shape_and_center()
    print("ok  test_face_fovea_shape_and_center")
    print("\n1 test passed")
