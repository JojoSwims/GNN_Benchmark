"""End-to-end tests for the benchmark pipeline using the LastValueModel toy model.

These tests validate the full pipeline:
    SyntheticLoader → DataWorkspace → sliding windows → temporal split
    → LastValueModel.fit() → LastValueModel.predict() → metrics

No network access is required; all data is generated in-memory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure the repo root is importable when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import BenchmarkRunner  # noqa: E402
from gnn_benchmark.core.types import DatasetInfo  # noqa: E402
from gnn_benchmark.datasets.base import DatasetLoader  # noqa: E402
from gnn_benchmark.exporters import SplitConfig, WindowConfig  # noqa: E402
from gnn_benchmark.models import LastValueModel  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset (no network access)
# ---------------------------------------------------------------------------

class SyntheticLoader(DatasetLoader):
    """Generates deterministic in-memory data for pipeline testing.

    Three nodes, 200 hourly timesteps.  Feature ``value[t] = float(t)`` for
    all nodes.  With a horizon of H the last-value model produces error = h
    at step h, so per-horizon MAEs are [1, 2, 3, …, H].
    """

    N_NODES = 3
    N_TIMESTAMPS = 200

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic",
            url="",
            frequency="1H",
            node_order=[f"n{i}" for i in range(self.N_NODES)],
            feature_columns=["value"],
        )

    def download_and_convert(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        timestamps = pd.date_range(
            "2024-01-01", periods=self.N_TIMESTAMPS, freq="1h"
        )
        rows = [
            {"ts": ts, "node_id": f"n{i}", "value": float(t)}
            for t, ts in enumerate(timestamps)
            for i in range(self.N_NODES)
        ]
        return pd.DataFrame(rows), None


# ---------------------------------------------------------------------------
# Unit tests — LastValueModel in isolation
# ---------------------------------------------------------------------------

class TestLastValueModel:
    """Unit tests that do not touch the pipeline at all."""

    def _fit(self, model: LastValueModel, S: int, L: int, H: int, N: int, D: int):
        x = torch.randn(S, L, N, D)
        y = torch.randn(S, H, N, D)
        history = model.fit(x, y, x, y, adj=None, config=None)
        return x, history

    def test_fit_returns_none(self):
        model = LastValueModel()
        _, history = self._fit(model, S=50, L=6, H=4, N=3, D=2)
        assert history is None

    def test_predict_shape(self):
        model = LastValueModel()
        S, L, H, N, D = 50, 6, 4, 3, 2
        self._fit(model, S=S, L=L, H=H, N=N, D=D)

        x_test = torch.randn(20, L, N, D)
        y_pred = model.predict(x_test, adj=None, config=None)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (20, H, N, D)

    def test_predict_repeats_last_value(self):
        """Every horizon step must equal x_test[:, -1, :, :]."""
        model = LastValueModel()
        S, L, H, N, D = 30, 8, 5, 2, 1
        self._fit(model, S=S, L=L, H=H, N=N, D=D)

        x_test = torch.randn(10, L, N, D)
        y_pred = model.predict(x_test, adj=None, config=None)

        last_values = x_test[:, -1, :, :].numpy()  # [10, N, D]
        for h in range(H):
            np.testing.assert_array_equal(
                y_pred[:, h, :, :],
                last_values,
                err_msg=f"Mismatch at horizon step {h}",
            )

    def test_predict_without_fit_raises(self):
        model = LastValueModel()
        x_test = torch.randn(5, 4, 3, 1)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(x_test, adj=None, config=None)

    def test_name(self):
        assert LastValueModel().name == "LastValue"

    def test_get_config_returns_dict(self):
        assert isinstance(LastValueModel().get_config(), dict)


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class TestE2EBenchmarkPipeline:
    """Full pipeline: SyntheticLoader → BenchmarkRunner → metrics."""

    INPUT_LENGTH = 6
    HORIZON = 6

    @pytest.fixture()
    def runner(self, tmp_path: Path) -> BenchmarkRunner:
        with patch.dict("benchmark.DATASET_REGISTRY", {"synthetic": SyntheticLoader}):
            yield BenchmarkRunner(
                workspace_dir=tmp_path / "ws",
                datasets=["synthetic"],
                window_config=WindowConfig(
                    input_length=self.INPUT_LENGTH,
                    horizon=self.HORIZON,
                ),
                split_config=SplitConfig(train_ratio=0.7, val_ratio=0.1),
                verbose=False,
            )

    def test_no_error(self, runner: BenchmarkRunner):
        result = runner.run(LastValueModel(), config=None)
        dr = result.dataset_results["synthetic"]
        assert dr.error is None, f"Pipeline raised an error: {dr.error}"

    def test_model_name_in_result(self, runner: BenchmarkRunner):
        result = runner.run(LastValueModel(), config=None)
        assert result.model_name == "LastValue"

    def test_metrics_are_finite(self, runner: BenchmarkRunner):
        result = runner.run(LastValueModel(), config=None)
        dr = result.dataset_results["synthetic"]
        assert np.isfinite(dr.mae),  f"MAE is not finite: {dr.mae}"
        assert np.isfinite(dr.rmse), f"RMSE is not finite: {dr.rmse}"
        assert np.isfinite(dr.mape), f"MAPE is not finite: {dr.mape}"

    def test_mae_is_positive(self, runner: BenchmarkRunner):
        """Persistence on linearly increasing data must have MAE > 0."""
        result = runner.run(LastValueModel(), config=None)
        dr = result.dataset_results["synthetic"]
        assert dr.mae > 0

    def test_per_horizon_count(self, runner: BenchmarkRunner):
        result = runner.run(LastValueModel(), config=None)
        dr = result.dataset_results["synthetic"]
        assert dr.per_horizon is not None
        assert len(dr.per_horizon) == self.HORIZON

    def test_per_horizon_mae_increases_with_step(self, runner: BenchmarkRunner):
        """On linear data (value[t]=t) the error at step h equals h, so
        per-horizon MAEs must be strictly increasing.
        """
        result = runner.run(LastValueModel(), config=None)
        dr = result.dataset_results["synthetic"]
        maes = [dr.per_horizon[h]["mae"] for h in range(1, self.HORIZON + 1)]
        for i in range(len(maes) - 1):
            assert maes[i] < maes[i + 1], (
                f"Expected per-horizon MAE to increase: step {i+1}={maes[i]:.4f} "
                f"≥ step {i+2}={maes[i+1]:.4f}"
            )

    def test_per_horizon_mae_values(self, runner: BenchmarkRunner):
        """On linear data the MAE at horizon step h equals exactly h."""
        result = runner.run(LastValueModel(), config=None)
        dr = result.dataset_results["synthetic"]
        for step in range(1, self.HORIZON + 1):
            expected = float(step)
            actual = dr.per_horizon[step]["mae"]
            assert abs(actual - expected) < 1e-4, (
                f"Step {step}: expected MAE={expected}, got {actual:.6f}"
            )

    def test_summary_runs_without_error(self, runner: BenchmarkRunner):
        result = runner.run(LastValueModel(), config=None)
        summary = result.summary()
        assert "LastValue" in summary
        assert "synthetic" in summary

    def test_to_dict_serialisable(self, runner: BenchmarkRunner):
        import json
        result = runner.run(LastValueModel(), config=None)
        data = result.to_dict()
        # Should serialise to JSON without error
        json.dumps(data)
        assert data["model_name"] == "LastValue"
        assert "synthetic" in data["datasets"]
