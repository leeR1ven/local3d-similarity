"""15,360-bit complementary encoding for a 16x16 RGB image."""
from __future__ import annotations

import numpy as np
import torch

IMG = 16
N_LEVELS = 10
IN_DIM = IMG * IMG * 3 * N_LEVELS * 2  # 15360
ACTIVE = IMG * IMG * 3 * N_LEVELS     # 7680
MAX_L1 = IMG * IMG * 3 * (N_LEVELS - 1)  # 6912


def encode(levels: torch.Tensor) -> torch.Tensor:
    """levels (B,16,16,3) in 0..9 -> (B,15360) with exactly 7680 ones."""
    t = torch.arange(N_LEVELS, device=levels.device)
    upper = (levels.unsqueeze(-1) > t).to(torch.float32)
    lower = 1.0 - upper
    return torch.stack([upper, lower], dim=-1).reshape(levels.shape[0], -1)


def encode_numpy(levels: np.ndarray) -> np.ndarray:
    t = np.arange(N_LEVELS)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], axis=-1).reshape(levels.shape[0], -1)
