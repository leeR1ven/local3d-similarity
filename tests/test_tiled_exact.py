"""Exact tiled activation must equal monolithic activation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.tiled_exact import DIMS, H, KS, W, exact_tiled
from threednet.local_patterns import encode
from threednet.model import Local3DNet


def test_exact_tiled_matches_monolithic():
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    levels = rng.integers(0, 10, (1, H, W, 3)).astype(np.int64)
    ok, a, b = exact_tiled(model, encode(levels))
    assert ok
    assert a == b


if __name__ == "__main__":
    test_exact_tiled_matches_monolithic()
    print("ok  test_exact_tiled_matches_monolithic")
    print("\n1 test passed")
