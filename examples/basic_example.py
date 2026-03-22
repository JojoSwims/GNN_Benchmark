#!/usr/bin/env python3
"""
Basic example: benchmark a simple model on one dataset.

This script demonstrates the full GNN Benchmark workflow:
1. Configure a BenchmarkRunner with a dataset and window parameters
2. Run a model (LastValueModel) through the benchmark
3. Print the evaluation results

Usage:
    python examples/basic_example.py
"""

import numpy as np

from benchmark import BenchmarkRunner
from gnn_benchmark import WindowConfig, SplitConfig, LastValueModel

# ── Run the built-in LastValueModel as a baseline ──────────────────────

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["elergone"],  # small dataset, no graph, fast to download
    window_config=WindowConfig(input_length=12, horizon=12),
    split_config=SplitConfig(train_ratio=0.7, val_ratio=0.1),
)

result = runner.run(LastValueModel())
print(result.summary())


# ── How to create your own model ──────────────────────────────────────
#
# Subclass BenchmarkModel and implement fit() and predict():
#
#     from gnn_benchmark import BenchmarkModel
#     from gnn_benchmark.models import TrainingHistory
#
#     class MeanModel(BenchmarkModel):
#         """Predicts the mean of each input window for all horizon steps."""
#
#         @property
#         def name(self) -> str:
#             return "MeanModel"
#
#         def fit(self, x_train, y_train, x_val, y_val, adj, config, norm_stats=None):
#             self._horizon = y_train.shape[1]
#             return None  # no training needed
#
#         def predict(self, x_test, adj, config):
#             # x_test: [S, L, N, D] torch.Tensor
#             x = x_test.numpy()
#             # Mean over the time dimension (axis=1), ignoring NaN
#             mean_val = np.nanmean(x, axis=1, keepdims=True)  # [S, 1, N, D]
#             return np.repeat(mean_val, self._horizon, axis=1)  # [S, H, N, D]
#
# Then run it:
#
#     result = runner.run(MeanModel())
#     print(result.summary())
