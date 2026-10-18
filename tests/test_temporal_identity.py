"""Recurrent state should be smoother than independent frame activations."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threednet.temporal_identity import cos, run_state


def test_state_is_smoother_than_frames():
    rng = np.random.default_rng(0)
    frames = (rng.random((8, 64)) < 0.5).astype(np.float32)
    states = run_state(frames, alpha=0.8)
    frame_cos = np.mean([cos(frames[i], frames[i + 1]) for i in range(7)])
    state_cos = np.mean([cos(states[i], states[i + 1]) for i in range(7)])
    assert state_cos > frame_cos


def test_state_keeps_memory_after_blank_gap():
    rng = np.random.default_rng(1)
    frames = (rng.random((4, 64)) < 0.5).astype(np.float32)
    before = run_state(frames)[-1]
    blank = np.zeros((3, 64), dtype=np.float32)
    states = run_state(np.concatenate([frames, blank], axis=0))
    after = states[-1]
    assert np.linalg.norm(after) > 0
    assert cos(before, after) > 0


if __name__ == "__main__":
    test_state_is_smoother_than_frames()
    test_state_keeps_memory_after_blank_gap()
    print("2 tests passed")
