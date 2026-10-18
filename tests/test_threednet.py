"""Checks for the fixed 3D locally-connected network."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from purenet.encoding import encode
from threednet.model import Local3DNet
from threednet.run import DIMS, KS


def test_3d_sparsity():
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    x = torch.randint(0, 10, (2, 16, 16, 3))
    hs = model(encode(x), hard=True)
    assert [int(h.sum(1)[0]) for h in hs] == [7680, 6144, 4608, 6144, 6144]
    assert [h.shape[1] for h in hs] == [15360, 15360, 15360, 30720, 61440]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
