"""Lightweight hyperparameter tuner for :class:`BenchmarkModel` submissions.

The tuner is model-agnostic: its logic is identical for every model.  Each
caller supplies the **search space** that matches its own config dataclass
— GWN tunes ``nhid``, MTGNN tunes ``conv_channels``, and so on — so the
set of hyperparameters varies per model, while the machinery does not.

Design rules
------------
- Selection is driven **only** by validation loss (min of
  :attr:`gnn_benchmark.models.TrainingHistory.val_loss`). The test set is
  never seen during tuning; the tuner does not call ``predict()`` and
  never touches ``x_test`` / ``y_test``.
- Data preparation happens **once**, via
  :meth:`gnn_benchmark.benchmark.BenchmarkRunner.prepare_dataset`. Each
  trial only pays for a fresh ``model.fit(...)`` call.
- The search-space API mirrors Optuna's ``trial.suggest_*`` shape, so a
  future ``OptunaTuner`` subclass can be dropped in without changing any
  existing call sites.

Typical usage::

    from gnn_benchmark.benchmark import BenchmarkRunner
    from gnn_benchmark.models import GWNConfig, GWNModel
    from gnn_benchmark.tuning import HyperparameterTuner, Categorical

    base = GWNConfig(max_epochs=10, batch_size=32, early_stop=5)

    tuner = HyperparameterTuner(
        model_factory=lambda: GWNModel(),
        base_config=base,
        dataset_key="beijing-air",
        workspace_dir="./benchmark_workspace",
        search_space={
            "lr":      Categorical([1e-3, 5e-4]),
            "dropout": Categorical([0.1, 0.3]),
        },
        strategy="grid",
    )
    tuning_result = tuner.run()
    print(tuning_result.summary())

    # Final, unbiased evaluation — uses the winning config exactly once.
    runner = BenchmarkRunner(
        workspace_dir="./benchmark_workspace",
        datasets=["beijing-air"],
    )
    final = runner.run(GWNModel(), config=tuning_result.best.config)
    print(final.summary())
"""

from __future__ import annotations

import gc
import itertools
import random
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from gnn_benchmark.benchmark import BenchmarkRunner, PreparedDataset
from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.models import BenchmarkModel, TrainingHistory
from gnn_benchmark.tuning.search_space import Sampler
from gnn_benchmark.utils.timing import WallTimer

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _free_trial_state(model: BenchmarkModel | None) -> None:
    """Release a trial's model and its cached GPU memory.

    Tuning loops accumulate VRAM because each fit() builds a fresh model
    + optimiser state + pinned DataLoader tensors and Python's GC does
    not run between trials. For long grids on large graphs (NY COVID
    with 3212 nodes is the worst offender) this is the difference
    between finishing the sweep and OOM-ing trial 4.
    """
    if model is not None:
        del model
    gc.collect()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        # ipc_collect is cheap and helps when workers pinned memory.
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Outcome of a single tuning trial.

    ``config`` is the full config passed to ``fit`` (base + overrides).
    ``overrides`` records only the fields the tuner varied, so results are
    easy to read in the summary table.

    ``duration_sec`` is total wall-clock; ``compute_time_sec`` is the
    pure-compute portion spent inside ``model.fit``, isolated so models
    with heavy graph construction or long DataLoader stalls can be
    compared on the actual training-step budget.
    """

    trial_index: int
    overrides: dict[str, Any]
    config: Any
    best_val_loss: float | None
    train_history: TrainingHistory | None
    duration_sec: float
    compute_time_sec: float = 0.0
    error: str | None = None


@dataclass
class TuningResult:
    """Aggregated output of :meth:`HyperparameterTuner.run`."""

    model_name: str
    dataset_name: str
    trials: list[TrialResult] = field(default_factory=list)
    best: TrialResult | None = None

    @property
    def total_compute_time_sec(self) -> float:
        """Sum of pure-compute fit time across all trials.

        Useful for forwarding to a downstream :class:`BenchmarkResult` so a
        results report can attribute the hyperparameter-search budget
        alongside the final training and inference budgets.
        """
        return sum(t.compute_time_sec for t in self.trials)

    def summary(self) -> str:
        """Return a formatted table of trials and the winning config."""
        lines: list[str] = []
        thick = "═" * 72
        thin = "─" * 72

        lines.append(f"\nTuning — model={self.model_name}  dataset={self.dataset_name}")
        lines.append(thick)
        if not self.trials:
            lines.append("(no trials run)")
            lines.append(thick)
            return "\n".join(lines)

        # Header
        field_names = sorted({k for t in self.trials for k in t.overrides})
        header = f"{'trial':<6} {'val_loss':>10}  "
        header += "  ".join(f"{n:>12}" for n in field_names)
        lines.append(header)
        lines.append(thin)

        for t in self.trials:
            if t.error is not None:
                row = f"{t.trial_index:<6} {'ERROR':>10}  "
            elif t.best_val_loss is None:
                row = f"{t.trial_index:<6} {'—':>10}  "
            else:
                row = f"{t.trial_index:<6} {t.best_val_loss:>10.4f}  "
            row += "  ".join(f"{str(t.overrides.get(n, '')):>12}" for n in field_names)
            if self.best is not None and t.trial_index == self.best.trial_index:
                row += "  ← best"
            if t.error is not None:
                row += f"  ERROR: {t.error}"
            lines.append(row)

        lines.append(thick)
        if self.best is not None:
            lines.append(f"Best: trial {self.best.trial_index}  "
                         f"val_loss={self.best.best_val_loss:.4f}  "
                         f"overrides={self.best.overrides}")
        # Compute-budget block: separates the "hyperparameter training
        # step" cost from the final training cost so reports can show
        # both numbers side by side.
        compute_total = self.total_compute_time_sec
        wall_total = sum(t.duration_sec for t in self.trials)
        if compute_total > 0 or wall_total > 0:
            lines.append(
                f"Hyperparameter search compute: "
                f"{compute_total:.2f}s of compute over "
                f"{wall_total:.2f}s wall ({len(self.trials)} trial(s))"
            )
        lines.append(thick)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "total_compute_time_sec": self.total_compute_time_sec,
            "trials": [
                {
                    "trial_index": t.trial_index,
                    "overrides": t.overrides,
                    "config": asdict(t.config) if is_dataclass(t.config) else None,
                    "best_val_loss": t.best_val_loss,
                    "train_loss": t.train_history.train_loss if t.train_history else None,
                    "val_loss": t.train_history.val_loss if t.train_history else None,
                    "duration_sec": t.duration_sec,
                    "compute_time_sec": t.compute_time_sec,
                    "error": t.error,
                }
                for t in self.trials
            ],
            "best_trial_index": self.best.trial_index if self.best else None,
        }


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

class HyperparameterTuner:
    """Grid / random search over a model's config dataclass.

    Args:
        model_factory:  Zero-arg callable returning a fresh
                        :class:`BenchmarkModel` per trial.
        base_config:    Config dataclass instance (e.g. ``GWNConfig(...)``).
                        Each trial clones this via ``dataclasses.replace``
                        and applies its overrides.
        dataset_key:    Key into ``BenchmarkRunner.DATASET_REGISTRY``.  A
                        single dataset per tuning run — keeps the control
                        loop simple and the signal (val loss) comparable
                        across trials.
        workspace_dir:  Passed to ``DataWorkspace``.
        search_space:   Mapping ``field_name -> Sampler``.  Field names must
                        exist on ``base_config`` — validated up front so
                        typos fail fast rather than silently ignoring a
                        misnamed knob.
        strategy:       ``"grid"`` (default) or ``"random"``.
        n_trials:       Required for random search; ignored for grid search
                        (which runs the full Cartesian product).
        seed:           Seeds the random sampler for reproducibility.
        verbose:        Print per-trial progress.
    """

    def __init__(
        self,
        model_factory: Callable[[], BenchmarkModel],
        base_config: Any,
        dataset_key: str,
        workspace_dir: str | Path = "./benchmark_workspace",
        search_space: Mapping[str, Sampler] | None = None,
        strategy: str = "grid",
        n_trials: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        if not is_dataclass(base_config):
            raise TypeError(
                "base_config must be a dataclass instance (e.g. GWNConfig)."
            )
        if strategy not in {"grid", "random"}:
            raise ValueError(f"strategy must be 'grid' or 'random', got {strategy!r}")
        if strategy == "random" and (n_trials is None or n_trials <= 0):
            raise ValueError("strategy='random' requires n_trials > 0.")

        search_space = dict(search_space or {})
        if not search_space:
            raise ValueError("search_space is empty — nothing to tune.")

        # Fail fast on typos: every field must exist on the dataclass.
        base_field_names = {f.name for f in base_config.__dataclass_fields__.values()}
        unknown = set(search_space) - base_field_names
        if unknown:
            raise ValueError(
                f"search_space references fields that are not on "
                f"{type(base_config).__name__}: {sorted(unknown)}"
            )

        self.model_factory = model_factory
        self.base_config = base_config
        self.dataset_key = dataset_key
        self.workspace_dir = Path(workspace_dir)
        self.search_space = search_space
        self.strategy = strategy
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------

    def _enumerate_trials(self) -> list[dict[str, Any]]:
        """Produce the list of override dicts for every trial, in order."""
        if self.strategy == "grid":
            names = list(self.search_space)
            value_lists = [self.search_space[n].grid_values() for n in names]
            return [
                dict(zip(names, combo))
                for combo in itertools.product(*value_lists)
            ]

        # random
        rng = random.Random(self.seed)
        trials: list[dict[str, Any]] = []
        for _ in range(self.n_trials or 0):
            trials.append(
                {name: sampler.sample(rng) for name, sampler in self.search_space.items()}
            )
        return trials

    # ------------------------------------------------------------------

    def run(self) -> TuningResult:
        """Run all trials and return a :class:`TuningResult`.

        The test set is **never** touched.  Each trial only invokes
        ``model.fit(x_train, y_train, x_val, y_val, adj, config)`` and
        reads ``min(TrainingHistory.val_loss)`` as the objective.
        """
        # Prepare data once — train/val/test splits are identical to what
        # BenchmarkRunner would produce for a final evaluation run.
        workspace = DataWorkspace(self.workspace_dir)
        runner = BenchmarkRunner(
            workspace_dir=self.workspace_dir,
            datasets=[self.dataset_key],
            verbose=self.verbose,
        )
        prepared: PreparedDataset = runner.prepare_dataset(workspace, self.dataset_key)

        trial_overrides = self._enumerate_trials()
        result = TuningResult(
            model_name=self._model_name(),
            dataset_name=self.dataset_key,
        )

        self._log(
            f"\n[tuner] {self._model_name()} on {self.dataset_key}: "
            f"{len(trial_overrides)} trial(s) via {self.strategy} search."
        )

        for idx, overrides in enumerate(trial_overrides):
            trial_cfg = replace(self.base_config, **overrides)
            self._log(f"[tuner] trial {idx + 1}/{len(trial_overrides)}  overrides={overrides}")

            start = time.time()
            history: TrainingHistory | None = None
            best_val: float | None = None
            error: str | None = None

            model: BenchmarkModel | None = None
            compute_elapsed = 0.0
            try:
                model = self.model_factory()
                with WallTimer() as timer:
                    history = model.fit(
                        prepared.x_train, prepared.y_train,
                        prepared.x_val, prepared.y_val,
                        prepared.adj, trial_cfg,
                    )
                # Prefer the wrapper's per-batch compute total when it
                # exposes one (the per-batch Stopwatch excludes
                # DataLoader iteration and host↔device transfer).  Fall
                # back to the outer wall-clock so legacy wrappers still
                # contribute a number.
                wrapper_compute = model.get_train_compute_sec()
                compute_elapsed = (
                    wrapper_compute if wrapper_compute is not None else timer.elapsed
                )
                if history is None:
                    raise RuntimeError(
                        "model.fit() returned None — the tuner needs a "
                        "TrainingHistory with per-epoch val_loss to score trials."
                    )
                if not history.val_loss:
                    raise RuntimeError(
                        "TrainingHistory.val_loss is empty; cannot score trial."
                    )
                best_val = float(min(history.val_loss))
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if self.verbose:
                    traceback.print_exc()
            finally:
                _free_trial_state(model)

            duration = time.time() - start
            trial = TrialResult(
                trial_index=idx,
                overrides=overrides,
                config=trial_cfg,
                best_val_loss=best_val,
                train_history=history,
                duration_sec=duration,
                compute_time_sec=compute_elapsed,
                error=error,
            )
            result.trials.append(trial)

            if error is not None:
                self._log(f"[tuner] trial {idx} FAILED — {error}")
            else:
                self._log(
                    f"[tuner] trial {idx} done  "
                    f"best_val_loss={best_val:.4f}  ({duration:.1f}s)"
                )

        # Pick winner: smallest finite val loss.
        valid = [t for t in result.trials if t.best_val_loss is not None]
        if valid:
            result.best = min(valid, key=lambda t: t.best_val_loss)  # type: ignore[arg-type]

        return result

    # ------------------------------------------------------------------

    def _model_name(self) -> str:
        try:
            return self.model_factory().name
        except Exception:  # noqa: BLE001
            return type(self.base_config).__name__.replace("Config", "")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
