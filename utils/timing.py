"""Compute-time measurement helper used by the benchmark harness.

The runner times the actual compute portions of training and inference
(``model.fit`` and ``model.predict``) so different models can be
compared on a per-call basis.  When CUDA is in use we synchronise
before reading the wall clock so kernel-launch lag does not bias the
measurement.
"""

from __future__ import annotations

import time

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class ComputeTimer:
    """Context manager that records elapsed compute time in seconds.

    Usage::

        with ComputeTimer() as t:
            model.fit(...)
        elapsed = t.elapsed
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "ComputeTimer":
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
