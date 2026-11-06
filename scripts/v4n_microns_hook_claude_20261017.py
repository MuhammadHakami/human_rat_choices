"""Picklable policy-input hook for the MICrONS-perception refit (Claude, 2026-10-17).

Replaces policy-input channels 0:144 (the pipeline's pooled camera vector) by the frozen 144 twin/rcore features of
the SAME frame. The pooled vector identifies its frame uniquely over the frozen frame set (20,820 frames, 0
collisions; verified in check). Environments, observer, source table and stored BC rows are untouched, so every
perception sees identical non-camera inputs and an identical (pixel-space) observer. All-zero camera rows (terminal
padding, never a rendered frame) stay zero. Any other unknown camera vector raises. The table is loaded lazily and
never pickled; the checkpoint stores only the perception name and the frozen hashes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "data/wacv/microns_perception_20261017"
ash = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class TwinSwap:
    def __init__(self, perc):
        d = json.loads((OUT / f"derive_{perc}.json").read_text())
        self.perc, self.keys_sha256, self.lookup_sha256 = perc, d["keys_sha256"], d["lookup_sha256"]
        self._map = None; self.calls = 0

    def __getstate__(self): return dict(perc=self.perc, keys_sha256=self.keys_sha256, lookup_sha256=self.lookup_sha256)

    def __setstate__(self, s): self.__dict__.update(s, _map=None, calls=0)

    def _load(self):
        K, Z = np.load(OUT / "keys.npy"), np.load(OUT / f"lookup_{self.perc}.npy")
        assert ash(K) == self.keys_sha256 and ash(Z) == self.lookup_sha256, "frozen MICrONS table changed"
        self._map = {k.tobytes(): z for k, z in zip(K, Z)}

    def __call__(self, module, inputs):
        if self._map is None: self._load()
        x = inputs[0]; cam = x[..., :144].detach().to("cpu", torch.float32).numpy().reshape(-1, 144)
        zero = np.zeros(144, np.float32)
        rows = [zero if not r.any() else self._map[r.tobytes()] for r in cam]     # KeyError = frame outside frozen set
        new = torch.as_tensor(np.stack(rows).reshape(x.shape[:-1] + (144,)), device=x.device, dtype=x.dtype)
        self.calls += 1
        return (torch.cat((new, x[..., 144:]), -1),) + tuple(inputs[1:])


def swap_of(net):
    return next((h for h in net.net.body._forward_pre_hooks.values() if isinstance(h, TwinSwap)), None)
