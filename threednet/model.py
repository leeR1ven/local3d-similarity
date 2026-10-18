"""A fixed 3D locally-connected network (no training).

Each layer is a 3D volume (x, y, z).  Every neuron has four inputs:
its own position (a copy) plus three random nearby neurons in the previous
volume.  The local offset is dx,dy in {-1,0,1} and dz in {-2..2}.  A fixed
random priority breaks top-k ties.

Layer widths are chosen so the exact active counts are
50/40/30/20/10%: 7,680 / 6,144 / 4,608 / 6,144 / 6,144.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from purenet.encoding import encode
from simnet.model import topk_mask


def _coords(dim: tuple[int, int, int]) -> np.ndarray:
    x, y, z = np.meshgrid(
        np.arange(dim[0]), np.arange(dim[1]), np.arange(dim[2]), indexing="ij"
    )
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


class Local3DNet(nn.Module):
    def __init__(
        self,
        dims=((16, 16, 60), (16, 16, 60), (16, 16, 60), (16, 16, 120), (16, 16, 240)),
        ks=(7680, 6144, 4608, 6144, 6144),
        fan_in: int = 4,
        alpha: float = 0.01,
        seed: int = 0,
        bias_scale: float = 0.5,
    ):
        super().__init__()
        self.dims = [tuple(d) for d in dims]
        self.ks = list(ks)
        self.fan_in = fan_in
        self.alpha = alpha
        rng = np.random.default_rng(seed)
        idxs, ws, biases = [], [], []
        for l in range(1, len(self.dims)):
            prev, out = self.dims[l - 1], self.dims[l]
            oc = _coords(out)
            n = len(oc)
            idx = np.zeros((n, fan_in), dtype=np.int64)
            w = np.zeros((n, fan_in), dtype=np.float32)
            # map output z onto the previous z range
            if out[2] > 1:
                zmap = np.round(oc[:, 2] * (prev[2] - 1) / (out[2] - 1)).astype(int)
            else:
                zmap = np.zeros(n, dtype=int)
            selfc = oc.copy()
            selfc[:, 2] = zmap
            selfc = np.clip(selfc, 0, np.array(prev) - 1)
            idx[:, 0] = np.ravel_multi_index(selfc.T, prev)
            w[:, 0] = 1.0
            for j in range(1, fan_in):
                nb = selfc.copy()
                nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
                nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
                nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
                idx[:, j] = np.ravel_multi_index(nb.T, prev)
                w[:, j] = (rng.random(n) * 0.5 + 0.5) * alpha
            bias = (rng.random(n) * bias_scale).astype(np.float32)
            idxs.append(torch.as_tensor(idx, dtype=torch.long))
            ws.append(torch.as_tensor(w, dtype=torch.float32))
            biases.append(torch.as_tensor(bias, dtype=torch.float32))

        # buffers move with .to(device) but are not trainable
        for l, (idx, w, b) in enumerate(zip(idxs, ws, biases)):
            self.register_buffer(f"idx_{l}", idx)
            self.register_buffer(f"w_{l}", w)
            self.register_buffer(f"b_{l}", b)
        self.n_layers = len(idxs)

    def forward(self, code: torch.Tensor, hard: bool = True):
        h = code
        hs = [h]
        for l in range(self.n_layers):
            idx = getattr(self, f"idx_{l}")
            w = getattr(self, f"w_{l}")
            b = getattr(self, f"b_{l}")
            s = (h[:, idx] * w.unsqueeze(0)).sum(-1) + b.unsqueeze(0)
            m = topk_mask(s, self.ks[l + 1])
            h = m if hard else m + s - s.detach()
            hs.append(h)
        return hs

    def forward_numpy(self, code: np.ndarray) -> list[np.ndarray]:
        if not hasattr(self, "_np_buffers"):
            self._np_buffers = [
                (getattr(self, f"idx_{l}").cpu().numpy(),
                 getattr(self, f"w_{l}").cpu().numpy(),
                 getattr(self, f"b_{l}").cpu().numpy())
                for l in range(self.n_layers)
            ]
        h = code.astype(np.float32)
        hs = [h]
        for l, (idx, w, b) in enumerate(self._np_buffers):
            s = (h[:, idx] * w[None, :, :]).sum(-1) + b[None, :]
            ix = np.argpartition(-s, self.ks[l + 1], axis=1)[:, : self.ks[l + 1]]
            o = np.zeros_like(s)
            np.put_along_axis(o, ix, 1.0, axis=1)
            h = o
            hs.append(h)
        return hs
