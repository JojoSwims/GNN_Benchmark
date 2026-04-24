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
    DivvyBikeshareLoader,
    ElergoneLoader,
    EULoadLoader,
    GDELTProtestLoader,
    LamaHCEDynamicLoader,
    MetroLALoader,
    MelPedsLoader,
    NOAABuoyLoader,
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
    "eu-load":              EULoadLoader,
    "nyiso":                NYISOLoader,
    "nyc-covid":            NYCovidLoader,
    "mel-peds":             MelPedsLoader,
    "lamah-ce-dynamic":     LamaHCEDynamicLoader,
    "noaa-buoy":            NOAABuoyLoader,
    "gdelt-protest":        GDELTProtestLoader,
    # Divvy Chicago bikeshare, 2024-03..2026-03 combined parquet.
    # Restricted to the Loop + Near North Side + Lincoln Park bbox
    # (N ≈ 515 stations). Static haversine edges sparsified with a
    # 2 km cutoff so message-passing sees a local neighborhood
    # instead of a nearly-uniform fully-connected graph.
    "divvy-bikeshare-static": lambda: DivvyBikeshareLoader(
        edges_mode="static",
        bbox=(41.87, 41.945, -87.67, -87.605),
        static_edge_cutoff_m=2000,
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PreparedDataset:
    """
    Windowed, split, tensorised tensors for a single dataset.

    Produced by :meth:`BenchmarkRunner.prepare_dataset` and consumed both
    by the runner itself (for ``fit`` + ``predict`` + metrics) and by the
    hyperparameter tuner (``fit`` only — the test tensors are ignored
    during tuning so the held-out set is never leaked).

    Attributes:
        dataset_name: Registry key of the dataset.
        x_train:      Training inputs  ``[S_train, seq_in_len, N, D_in]``,
                      NaN = missing.
        y_train:      Training targets ``[S_train, seq_out_len, N, D_out]``.
        x_val:        Validation inputs.
        y_val:        Validation targets.
        x_test:       Test inputs (used only for final evaluation).
        y_test:       Test targets as a raw numpy array (used only for
                      final metric computation).
        adj:          Adjacency matrix ``[N, N]`` or ``None``.
        window_config: The ``WindowConfig`` the splits were built with.
        split_config:  The ``SplitConfig`` the splits were built with.
    """

    dataset_name: str
    x_train: "torch.Tensor"
    y_train: "torch.Tensor"
    x_val: "torch.Tensor"
    y_val: "torch.Tensor"
    x_test: "torch.Tensor"
    y_test: np.ndarray
    adj: np.ndarray | None
    window_config: Any
    split_config: Any


@dataclass
class DatasetResult:
    """
    Benchmark metrics for a single dataset.

    Attributes:
        dataset_name: Registry key of the dataset.
        mae:          Mean Absolute Error (original units).
        rmse:         Root Mean Squared Error (original units).
        mape:         Mean Absolute Percentage Error (0–100 scale).
        per_horizon:  Per-step metrics ``{step: {"mae": ..., "rmse": ..., "mape": ...}}``
                      for every horizon step h=1..H.
        error:        Exception message if this dataset failed; other fields are ``None``.
    """

    dataset_name: str
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    per_horizon: dict[int, dict[str, float]] = field(default_factory=dict)
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

    # Horizons highlighted in the per-dataset table
    _DISPLAY_HORIZONS = (1, 3, 6, 12)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """
        Return a formatted table of results.

        For each dataset a section is printed showing metrics at horizon steps
        h=1, h=3, h=6, h=12 (whichever exist) plus an ``Avg`` row (mean across
        all horizon steps).  Datasets that failed are shown with ``ERROR``.
        A final section shows the mean of ``Avg`` across all successful datasets.
        """
        lines: list[str] = []
        thick = "═" * 56
        thin  = "─" * 56

        lines.append(f"\nModel: {self.model_name}")

        mae_vals, rmse_vals, mape_vals = [], [], []

        for name, r in self.dataset_results.items():
            lines.append(thick)
            if r.error:
                lines.append(f"Dataset: {name}  ERROR: {r.error}")
                continue

            lines.append(f"Dataset: {name}")
            lines.append(thin)
            lines.append(f"{'Horizon':<10} {'MAE':>9} {'RMSE':>9} {'MAPE':>9}")
            lines.append(thin)

            # Rows for specific horizons that exist in this result
            for h in self._DISPLAY_HORIZONS:
                if h in r.per_horizon:
                    hm = r.per_horizon[h]
                    lines.append(
                        f"{'h='+str(h):<10} {hm['mae']:>9.4f}"
                        f" {hm['rmse']:>9.4f} {hm['mape']:>8.2f}%"
                    )

            # Avg row (mean of ALL horizon steps)
            lines.append(thin)
            lines.append(
                f"{'Avg':<10} {r.mae:>9.4f} {r.rmse:>9.4f} {r.mape:>8.2f}%"
            )

            mae_vals.append(r.mae)
            rmse_vals.append(r.rmse)
            mape_vals.append(r.mape)

        # Cross-dataset mean
        if mae_vals:
            lines.append(thick)
            lines.append("Mean across datasets")
            lines.append(thin)
            lines.append(f"{'Horizon':<10} {'MAE':>9} {'RMSE':>9} {'MAPE':>9}")
            lines.append(thin)
            lines.append(
                f"{'Avg':<10} {np.mean(mae_vals):>9.4f}"
                f" {np.mean(rmse_vals):>9.4f}"
                f" {np.mean(mape_vals):>8.2f}%"
            )

        lines.append(thick)
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

    def prepare_dataset(
        self,
        workspace: DataWorkspace,
        dataset_key: str,
    ) -> PreparedDataset:
        """
        Download, window, split, and tensorise a single dataset.

        This is the deterministic data-prep portion of the benchmark pipeline
        extracted into a reusable helper.  Both ``_run_dataset`` and the
        hyperparameter tuner call it; the splits they see are therefore
        guaranteed to be identical.

        The returned :class:`PreparedDataset` carries all six tensors plus
        the adjacency and config metadata.  Consumers that should not see the
        test set (i.e. the tuner) simply ignore ``x_test`` / ``y_test``.
        """
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset: {dataset_key!r}. "
                f"Available: {list(DATASET_REGISTRY)}"
            )

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

        return PreparedDataset(
            dataset_name=dataset_key,
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val=y_val_t,
            x_test=x_test_t,
            y_test=y_test,
            adj=adj,
            window_config=wc,
            split_config=sc,
        )

    def _run_dataset(
        self,
        workspace: DataWorkspace,
        dataset_key: str,
        model: BenchmarkModel,
        config: Any,
    ) -> DatasetResult:
        prepared = self.prepare_dataset(workspace, dataset_key)

        # 7. Fit
        self._log(f"[{dataset_key}] Fitting model …")
        model.fit(
            prepared.x_train, prepared.y_train,
            prepared.x_val, prepared.y_val,
            prepared.adj, config,
        )

        # 8. Predict
        self._log(f"[{dataset_key}] Predicting …")
        y_pred = model.predict(prepared.x_test, prepared.adj, config)
        y_pred = np.asarray(y_pred)

        y_test = prepared.y_test

        # Validate output shape
        expected = y_test.shape
        if y_pred.shape != expected:
            raise ValueError(
                f"predict() returned shape {y_pred.shape}, expected {expected}. "
                "y_pred must be [num_test_samples, seq_out_len, N, D_out]."
            )

        # 9. Per-horizon metrics (always computed)
        horizon = prepared.window_config.horizon
        per_horizon: dict[int, dict[str, float]] = {}
        for h in range(horizon):
            yt_h = y_test[:, h, :, :]
            yp_h = y_pred[:, h, :, :]
            mask_h = ~np.isnan(yt_h) & ~np.isnan(yp_h)
            per_horizon[h + 1] = {
                "mae":  mae(yt_h,  yp_h, mask=mask_h),
                "rmse": rmse(yt_h, yp_h, mask=mask_h),
                "mape": mape(yt_h, yp_h, mask=mask_h),
            }

        # 10. Aggregate = mean of per-horizon metrics (matches "Avg" row in summary)
        dataset_mae  = float(np.mean([v["mae"]  for v in per_horizon.values()]))
        dataset_rmse = float(np.mean([v["rmse"] for v in per_horizon.values()]))
        dataset_mape = float(np.mean([v["mape"] for v in per_horizon.values()]))

        return DatasetResult(
            dataset_name=dataset_key,
            mae=dataset_mae,
            rmse=dataset_rmse,
            mape=dataset_mape,
            per_horizon=per_horizon,  # always populated (h=1..horizon)
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
