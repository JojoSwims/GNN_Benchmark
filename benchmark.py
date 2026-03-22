"""
Full end-to-end benchmark runner for GNN model submissions.

Usage
-----
::

    from benchmark import BenchmarkRunner
    from gnn_benchmark.models import BenchmarkModel, TrainingHistory

    class MyGNN(BenchmarkModel):
        name = "MyGNN"

        def fit(self, x_train, y_train, x_val, y_val, adj, config):
            # x_train: (S_train, L, N, D_in) torch.Tensor — NaN = missing
            # Create your own DataLoader, normalise as you see fit:
            #   from torch.utils.data import DataLoader, TensorDataset
            #   loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64)
            return None

        def predict(self, x_test, adj, config):
            # x_test: (S_test, L, N, D_in) torch.Tensor
            # Return np.ndarray [S_test, seq_out_len, N, D_out] in original units
            ...

    runner = BenchmarkRunner(
        workspace_dir="./benchmark_workspace",
        datasets=["metr-la", "pems-bay"],        # None = all datasets
    )
    result = runner.run(MyGNN(), config=my_config)
    print(result.summary())

Pipeline (per dataset)
----------------------
1.  Prepare IR  — download and cache via ``DataWorkspace``
2.  Window      — sliding windows → ``(x, y)`` arrays via ``create_sliding_windows``
3.  Split       — temporal train / val / test via ``split_by_time``
4.  Tensors     — convert to float32 ``torch.Tensor`` (NaN preserved for missing values)
5.  Fit         — ``model.fit(x_train, y_train, x_val, y_val, adj, config)``
6.  Predict     — ``model.predict(x_test, adj, config)``
7.  Metrics     — MAE, RMSE, MAPE in original units; NaN positions excluded

Notes
-----
- Normalisation is the model's responsibility.
- Missing values are represented as ``float('nan')`` in the tensors — zeros
  are **not** used as sentinels.
- ``adj`` is ``None`` for datasets that carry no edge information.
- ``config`` is passed through unchanged; the runner does not inspect it.
- Window and split configurations are fixed per dataset via ``DatasetInfo``.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.datasets import (
    BeijingAirLoader,
    Cluster1AirLoader,
    Cluster2AirLoader,
    ElergoneLoader,
    MetroLALoader,
    MelPedsLoader,
    NYCovidLoader,
    NYISOLoader,
    PEMS04Loader,
    PEMS08Loader,
    PEMSBayLoader,
)
from gnn_benchmark.core.types import WindowConfig, SplitConfig  # noqa: F401 (re-exported)
from gnn_benchmark.utils.data import (
    create_sliding_windows,
    split_by_time,
)
from gnn_benchmark.models import BenchmarkModel
from gnn_benchmark.utils.metrics import mae, mape, rmse


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, Any] = {
    "metr-la":              MetroLALoader,
    "pems-bay":             PEMSBayLoader,
    "pems04":               PEMS04Loader,
    "pems08":               PEMS08Loader,
    "beijing-air":          BeijingAirLoader,
    "beijing-air-cluster1": Cluster1AirLoader,
    "beijing-air-cluster2": Cluster2AirLoader,
    "elergone":             ElergoneLoader,
    "nyiso":                NYISOLoader,
    "nyc-covid":            NYCovidLoader,
    "mel-peds":             MelPedsLoader,
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
        mae:          Mean Absolute Error (original units).
        rmse:         Root Mean Squared Error (original units).
        mape:         Mean Absolute Percentage Error (0–100 scale).
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
        lines.append("(metrics in original units)")
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

    Window and split configurations are fixed per dataset (defined in
    ``DatasetInfo.window_config`` and ``DatasetInfo.split_config``).

    Args:
        workspace_dir: Directory used by ``DataWorkspace`` for caching.
                       Created automatically if it does not exist.
        datasets:      List of dataset keys from ``DATASET_REGISTRY`` to
                       benchmark against.  ``None`` runs all registered datasets.
        verbose:       Print progress messages when ``True``.
    """

    def __init__(
        self,
        workspace_dir: str | Path = "./benchmark_workspace",
        datasets: list[str] | None = None,
        verbose: bool = True,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
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
        2. Creates sliding windows and temporal splits.
        3. Converts splits to float32 torch.Tensor (NaN preserved for missing values).
        4. Calls ``model.fit(x_train, y_train, x_val, y_val, adj, config)``.
        5. Calls ``model.predict(x_test, adj, config)``.
        6. Computes MAE, RMSE, MAPE in original units (NaN positions in y_test excluded).

        Dataset failures are caught and recorded in ``DatasetResult.error``
        so that a single broken dataset does not abort the full run.

        Args:
            model:  A fitted or unfitted ``BenchmarkModel`` instance.
            config: Opaque config object forwarded to ``fit`` and ``predict``.

        Returns:
            ``BenchmarkResult`` with per-dataset and aggregate metrics.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for BenchmarkRunner. "
                "Install it with: pip install torch"
            )

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
        # 1. Prepare IR (raw data, NaN for missing values — no imputation applied)
        loader_cls = DATASET_REGISTRY[dataset_key]
        loader = loader_cls()
        self._log(f"[{dataset_key}] Preparing data …")
        ir = loader.prepare(workspace)

        # Window and split configs are fixed per dataset
        wc = loader.info.window_config
        sc = loader.info.split_config

        # 2. Build tensor and select columns
        x_cols = wc.input_columns  # None → all features
        y_cols = wc.target_columns or x_cols

        x_data = ir.to_tensor(x_cols)  # (T, N, D_in)  — may contain NaN
        y_data = ir.to_tensor(y_cols)  # (T, N, D_out) — may contain NaN

        # 3. Sliding windows
        self._log(f"[{dataset_key}] Windowing …")
        x_all, _ = create_sliding_windows(x_data, wc.input_length, wc.horizon, wc.y_start)
        _, y_all = create_sliding_windows(y_data, wc.input_length, wc.horizon, wc.y_start)

        # 4. Temporal split
        x_train, x_val, x_test = split_by_time(x_all, sc.train_ratio, sc.val_ratio)
        y_train, y_val, y_test = split_by_time(y_all, sc.train_ratio, sc.val_ratio)

        # 5. Convert to float32 tensors (NaN values preserved in float32)
        x_train_t = torch.from_numpy(x_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        x_val_t   = torch.from_numpy(x_val.astype(np.float32))
        y_val_t   = torch.from_numpy(y_val.astype(np.float32))
        x_test_t  = torch.from_numpy(x_test.astype(np.float32))

        # 6. Adjacency
        adj = ir.get_adjacency_matrix() if ir.edges is not None else None

        # 7. Fit
        self._log(f"[{dataset_key}] Fitting model …")
        model.fit(x_train_t, y_train_t, x_val_t, y_val_t, adj, config)

        # 8. Predict
        self._log(f"[{dataset_key}] Predicting …")
        y_pred = model.predict(x_test_t, adj, config)
        y_pred = np.asarray(y_pred)

        # Validate output shape
        expected = y_test.shape
        if y_pred.shape != expected:
            raise ValueError(
                f"predict() returned shape {y_pred.shape}, expected {expected}. "
                "y_pred must be [num_test_samples, seq_out_len, N, D_out]."
            )

        # 9. Metrics (original units; exclude NaN in ground truth or predictions)
        valid_mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
        dataset_mae  = mae(y_test,  y_pred, mask=valid_mask)
        dataset_rmse = rmse(y_test, y_pred, mask=valid_mask)
        dataset_mape = mape(y_test, y_pred, mask=valid_mask)

        # 10. Per-horizon metrics
        horizon = wc.horizon
        per_horizon: dict[int, dict[str, float]] | None = None
        if horizon > 1:
            per_horizon = {}
            for h in range(horizon):
                yt_h = y_test[:, h, :, :]
                yp_h = y_pred[:, h, :, :]
                mask_h = ~np.isnan(yt_h) & ~np.isnan(yp_h)
                per_horizon[h + 1] = {
                    "mae":  mae(yt_h,  yp_h, mask=mask_h),
                    "rmse": rmse(yt_h, yp_h, mask=mask_h),
                    "mape": mape(yt_h, yp_h, mask=mask_h),
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
