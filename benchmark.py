"""
Full end-to-end benchmark runner for GNN model submissions.

Usage
-----
::

    from benchmark import BenchmarkRunner
    from gnn_benchmark.models import BenchmarkModel, TrainingHistory
    from gnn_benchmark.exporters import WindowConfig, SplitConfig

    class MyGNN(BenchmarkModel):
        name = "MyGNN"

        def fit(self, train_loader, val_loader, adj, config):
            ...  # train your model
            return None

        def predict(self, test_loader, adj, config):
            ...  # return np.ndarray [num_test_samples, seq_out_len, N, D_out]

    runner = BenchmarkRunner(
        workspace_dir="./benchmark_workspace",
        datasets=["metr-la", "pems-bay"],        # None = all datasets
        window_config=WindowConfig(input_length=12, horizon=12),
    )
    result = runner.run(MyGNN(), config=my_config)
    print(result.summary())

Pipeline (per dataset)
----------------------
1.  Prepare IR  — download and cache via ``DataWorkspace``
2.  Transform   — ``FillZeros`` then ``ZScoreNormalize`` (train split only)
3.  Window      — sliding windows → ``(x, y)`` arrays via ``create_sliding_windows``
4.  Split       — temporal train / val / test via ``split_by_time``
5.  DataLoaders — ``make_dataloaders`` (test loader always ``shuffle=False``)
6.  Fit         — ``model.fit(train_loader, val_loader, adj, config)``
7.  Predict     — ``model.predict(test_loader, adj, config)``
8.  Metrics     — MAE, RMSE, MAPE in normalised space

Notes
-----
- Metrics are computed in the **normalised space** (after ``ZScoreNormalize``).
  Raw-unit metrics can be derived by the caller using ``ir.metadata.extra``
  (keys ``zscore_means`` / ``zscore_stds``).
- ``adj`` is ``None`` for datasets that carry no edge information.
- ``config`` is passed through unchanged; the runner does not inspect it.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.datasets import (
    BeijingAirLoader,
    ElergoneLoader,
    MetroLALoader,
    MelPedsLoader,
    NYCovidLoader,
    NYISOLoader,
    PEMS04Loader,
    PEMS08Loader,
    PEMSBayLoader,
)
from gnn_benchmark.exporters import WindowConfig, SplitConfig
from gnn_benchmark.exporters.dataloader import (
    create_sliding_windows,
    make_dataloaders,
    split_by_time,
)
from gnn_benchmark.models import BenchmarkModel
from gnn_benchmark.transforms import FillZeros, ZScoreNormalize
from gnn_benchmark.utils.metrics import mae, mape, rmse


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, Any] = {
    "metr-la":    MetroLALoader,
    "pems-bay":   PEMSBayLoader,
    "pems04":     PEMS04Loader,
    "pems08":     PEMS08Loader,
    "beijing-air": BeijingAirLoader,
    "elergone":   ElergoneLoader,
    "nyiso":      NYISOLoader,
    "nyc-covid":  NYCovidLoader,
    "mel-peds":   MelPedsLoader,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    """
    Benchmark metrics for a single dataset.

    Attributes:
        dataset_name: Registry key of the dataset.
        mae:          Mean Absolute Error (normalised space).
        rmse:         Root Mean Squared Error (normalised space).
        mape:         Mean Absolute Percentage Error (normalised space, 0–100 scale).
        per_horizon:  Per-step metrics ``{step: {"mae": ..., "rmse": ..., "mape": ...}}``.
                      ``None`` when ``horizon == 1``.
        error:        Exception message if this dataset failed; other fields are ``None``.
    """

    dataset_name: str
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    per_horizon: dict[int, dict[str, float]] | None = None
    error: str | None = None


@dataclass
class BenchmarkResult:
    """
    Aggregated results returned by ``BenchmarkRunner.run()``.

    Attributes:
        model_name:      Value of ``BenchmarkModel.name``.
        model_config:    Dict returned by ``BenchmarkModel.get_config()``.
        dataset_results: Mapping from dataset key to ``DatasetResult``.
    """

    model_name: str
    model_config: dict = field(default_factory=dict)
    dataset_results: dict[str, DatasetResult] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """
        Return a formatted table of results.

        Datasets that failed are shown with ``ERROR`` instead of metric values.
        An aggregate row shows the mean across successful datasets.
        """
        lines: list[str] = []
        sep = "─" * 56
        lines.append(f"\nModel: {self.model_name}")
        lines.append(sep)
        lines.append(f"{'Dataset':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
        lines.append(sep)

        mae_vals, rmse_vals, mape_vals = [], [], []

        for name, r in self.dataset_results.items():
            if r.error:
                lines.append(f"{name:<20} {'ERROR':>8}   {r.error[:24]}")
            else:
                lines.append(
                    f"{name:<20} {r.mae:>8.4f} {r.rmse:>8.4f} {r.mape:>7.2f}%"
                )
                mae_vals.append(r.mae)
                rmse_vals.append(r.rmse)
                mape_vals.append(r.mape)

        if mae_vals:
            lines.append(sep)
            lines.append(
                f"{'Mean':<20} {np.mean(mae_vals):>8.4f}"
                f" {np.mean(rmse_vals):>8.4f}"
                f" {np.mean(mape_vals):>7.2f}%"
            )

        lines.append(sep)
        lines.append("(metrics in normalised space)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "datasets": {
                name: {
                    "mae": r.mae,
                    "rmse": r.rmse,
                    "mape": r.mape,
                    "per_horizon": r.per_horizon,
                    "error": r.error,
                }
                for name, r in self.dataset_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Drives the full benchmark pipeline for one or more datasets.

    Args:
        workspace_dir: Directory used by ``DataWorkspace`` for caching.
                       Created automatically if it does not exist.
        datasets:      List of dataset keys from ``DATASET_REGISTRY`` to
                       benchmark against.  ``None`` runs all registered datasets.
        window_config: Sliding window parameters (input length, horizon, …).
        split_config:  Temporal train / val / test ratios.
        batch_size:    Batch size forwarded to ``make_dataloaders``.
        verbose:       Print progress messages when ``True``.
    """

    def __init__(
        self,
        workspace_dir: str | Path = "./benchmark_workspace",
        datasets: list[str] | None = None,
        window_config: WindowConfig | None = None,
        split_config: SplitConfig | None = None,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.window_config = window_config or WindowConfig()
        self.split_config = split_config or SplitConfig()
        self.batch_size = batch_size
        self.verbose = verbose

        if datasets is None:
            self.datasets = list(DATASET_REGISTRY.keys())
        else:
            unknown = set(datasets) - set(DATASET_REGISTRY)
            if unknown:
                raise ValueError(
                    f"Unknown dataset(s): {unknown}. "
                    f"Available: {list(DATASET_REGISTRY)}"
                )
            self.datasets = datasets

    # ------------------------------------------------------------------

    def run(
        self,
        model: BenchmarkModel,
        config: Any = None,
    ) -> BenchmarkResult:
        """
        Run the full benchmark for all configured datasets.

        For each dataset the runner:
        1. Prepares the IR (download + cache).
        2. Applies ``FillZeros`` and ``ZScoreNormalize`` (train-split stats).
        3. Creates sliding windows and temporal splits.
        4. Builds DataLoaders (test loader always ``shuffle=False``).
        5. Calls ``model.fit(train_loader, val_loader, adj, config)``.
        6. Calls ``model.predict(test_loader, adj, config)``.
        7. Computes MAE, RMSE, MAPE and per-horizon metrics.

        Dataset failures are caught and recorded in ``DatasetResult.error``
        so that a single broken dataset does not abort the full run.

        Args:
            model:  A fitted or unfitted ``BenchmarkModel`` instance.
            config: Opaque config object forwarded to ``fit`` and ``predict``.

        Returns:
            ``BenchmarkResult`` with per-dataset and aggregate metrics.
        """
        result = BenchmarkResult(
            model_name=model.name,
            model_config=model.get_config(),
        )

        workspace = DataWorkspace(self.workspace_dir)

        for dataset_key in self.datasets:
            self._log(f"\n[{dataset_key}] Starting …")
            try:
                dr = self._run_dataset(workspace, dataset_key, model, config)
            except Exception as exc:  # noqa: BLE001
                dr = DatasetResult(
                    dataset_name=dataset_key,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if self.verbose:
                    traceback.print_exc()

            result.dataset_results[dataset_key] = dr
            if dr.error:
                self._log(f"[{dataset_key}] FAILED — {dr.error}")
            else:
                self._log(
                    f"[{dataset_key}] Done  "
                    f"MAE={dr.mae:.4f}  RMSE={dr.rmse:.4f}  MAPE={dr.mape:.2f}%"
                )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dataset(
        self,
        workspace: DataWorkspace,
        dataset_key: str,
        model: BenchmarkModel,
        config: Any,
    ) -> DatasetResult:
        wc = self.window_config
        sc = self.split_config

        # 1. Prepare IR
        loader_cls = DATASET_REGISTRY[dataset_key]
        loader = loader_cls()
        self._log(f"[{dataset_key}] Preparing data …")
        ir = loader.prepare(workspace)

        # 2. Standard transforms
        self._log(f"[{dataset_key}] Applying transforms …")
        train_end, _ = ir.get_split_timestamps(sc.train_ratio, sc.val_ratio)
        ir.apply(FillZeros())
        ir.apply(ZScoreNormalize(train_end=train_end))

        # 3. Build tensor and select columns
        x_cols = wc.input_columns  # None → all features
        y_cols = wc.target_columns or x_cols

        x_data = ir.to_tensor(x_cols)  # (T, N, D_in)
        y_data = ir.to_tensor(y_cols)  # (T, N, D_out)

        # 4. Sliding windows
        self._log(f"[{dataset_key}] Windowing …")
        x_all, _ = create_sliding_windows(x_data, wc.input_length, wc.horizon, wc.y_start)
        _, y_all = create_sliding_windows(y_data, wc.input_length, wc.horizon, wc.y_start)

        # 5. Temporal split
        x_train, x_val, x_test = split_by_time(x_all, sc.train_ratio, sc.val_ratio)
        y_train, y_val, y_test = split_by_time(y_all, sc.train_ratio, sc.val_ratio)

        # 6. DataLoaders
        train_loader, val_loader, test_loader = make_dataloaders(
            x_train, y_train,
            x_val,   y_val,
            x_test,  y_test,
            batch_size=self.batch_size,
        )

        # 7. Adjacency
        adj = ir.get_adjacency_matrix() if ir.edges is not None else None

        # 8. Fit
        self._log(f"[{dataset_key}] Fitting model …")
        model.fit(train_loader, val_loader, adj, config)

        # 9. Predict
        self._log(f"[{dataset_key}] Predicting …")
        y_pred = model.predict(test_loader, adj, config)
        y_pred = np.asarray(y_pred)

        # Validate output shape
        expected = y_test.shape
        if y_pred.shape != expected:
            raise ValueError(
                f"predict() returned shape {y_pred.shape}, expected {expected}. "
                "y_pred must be [num_test_samples, seq_out_len, N, D_out]."
            )

        # 10. Metrics (overall)
        dataset_mae  = mae(y_test,  y_pred)
        dataset_rmse = rmse(y_test, y_pred)
        dataset_mape = mape(y_test, y_pred)

        # 11. Per-horizon metrics
        horizon = wc.horizon
        per_horizon: dict[int, dict[str, float]] | None = None
        if horizon > 1:
            per_horizon = {}
            for h in range(horizon):
                yt_h = y_test[:, h, :, :]
                yp_h = y_pred[:, h, :, :]
                per_horizon[h + 1] = {
                    "mae":  mae(yt_h,  yp_h),
                    "rmse": rmse(yt_h, yp_h),
                    "mape": mape(yt_h, yp_h),
                }

        return DatasetResult(
            dataset_name=dataset_key,
            mae=dataset_mae,
            rmse=dataset_rmse,
            mape=dataset_mape,
            per_horizon=per_horizon,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
