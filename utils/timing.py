"""Wall-clock and compute-time measurement helpers used by the benchmark.

Two helpers live here:

* :class:`WallTimer` (alias :class:`ComputeTimer` for backwards compatibility)
  is a one-shot context manager that records wall-clock seconds spent in the
  block, syncing CUDA on enter and exit.  The benchmark harness uses it to
  measure end-to-end ``model.fit`` / ``model.predict`` time.

* :class:`Stopwatch` is an *accumulator* — it can be entered repeatedly and
  totals the time spent inside it.  Model wrappers use it to time only the
  pure compute spans (forward + backward + step + criterion + scaler ops),
  excluding DataLoader iteration, host→device transfers, and best-state
  CPU clones.  That way the published per-model compute budget is not
  inflated by data movement that has nothing to do with the model itself.

When CUDA is in use both helpers call ``torch.cuda.synchronize()`` so
asynchronous kernel launches do not leak into the next measurement.
"""

from __future__ import annotations

import time

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _cuda_sync() -> None:
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()


class WallTimer:
    """One-shot context manager recording elapsed wall-clock seconds.

    Usage::

        with WallTimer() as t:
            model.fit(...)
        elapsed = t.elapsed
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "WallTimer":
        _cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _cuda_sync()
        self.elapsed = time.perf_counter() - self._start


# Backwards-compatible alias — older code imports ComputeTimer.
ComputeTimer = WallTimer


class Stopwatch:
    """Accumulator that totals wall-clock seconds spent inside it.

    Unlike :class:`WallTimer`, a :class:`Stopwatch` is meant to be
    re-entered many times in a single training loop — once per batch,
    say — so the cumulative compute span can be reported separately
    from data-loading and bookkeeping time.  Both the context-manager
    and ``start()`` / ``stop()`` interfaces are supported::

        sw = Stopwatch()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)        # NOT timed
            with sw:
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
        total_compute_sec = sw.elapsed

    The helper is reset by calling :meth:`reset`.  CUDA syncs bracket
    every entered span so an unfinished kernel from the previous batch
    does not bleed into the next measurement.
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: float | None = None

    def reset(self) -> None:
        self.elapsed = 0.0
        self._t0 = None

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("Stopwatch.start() called while already running.")
        _cuda_sync()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is None:
            raise RuntimeError("Stopwatch.stop() called without a matching start().")
        _cuda_sync()
        self.elapsed += time.perf_counter() - self._t0
        self._t0 = None

    def __enter__(self) -> "Stopwatch":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
