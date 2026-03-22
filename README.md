# GNN Benchmark

A Python toolkit for benchmarking Graph Neural Networks on spatiotemporal time series data.

## Overview

GNN Benchmark provides a streamlined pipeline for:

1. **Downloading datasets** from external sources (traffic, air quality, energy, etc.)
2. **Preparing data** as sliding-window `(x, y)` train/val/test tensors
3. **Defining a model contract** — what models receive and what they return
4. **Evaluating predictions** against ground truth (MAE, RMSE, MAPE)

## Installation

```bash
git clone https://github.com/JojoSwims/GNN_Benchmark.git
cd GNN_Benchmark

pip install -r requirements.txt
pip install torch       # Required for BenchmarkRunner
pip install gdown       # Optional: for PEMS-BAY and METR-LA datasets
```

## Quick Start

```python
from benchmark import BenchmarkRunner
from gnn_benchmark import LastValueModel

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["metr-la"],
)

result = runner.run(LastValueModel())
print(result.summary())
```

See [`examples/basic_example.py`](examples/basic_example.py) for a runnable example.

## Model Contract

Subclass `BenchmarkModel` and implement `fit` and `predict`:

```python
from gnn_benchmark.models import BenchmarkModel

class MyGNN(BenchmarkModel):
    @property
    def name(self) -> str:
        return "MyGNN"

    def fit(self, x_train, y_train, x_val, y_val, adj, config):
        # x_train: [S, L, N, D_in] torch.Tensor — NaN = missing
        # y_train: [S, H, N, D_out] torch.Tensor
        # adj: [N, N] numpy array or None (no graph)
        ...
        return None  # or TrainingHistory(train_loss=[...], val_loss=[...])

    def predict(self, x_test, adj, config):
        # x_test: [S_test, L, N, D_in] torch.Tensor
        # Return: np.ndarray [S_test, H, N, D_out] in original (unnormalised) units
        ...
```

### Key rules

- Normalisation is the model's responsibility
- Missing values are `NaN`, not zero
- `adj` is `None` for datasets without graph structure
- `y_pred` must be in **original units** (metrics are computed against raw ground truth)
- Models handle their own batching (no DataLoader is created by the harness)

## Available Datasets

| Key | Dataset | Nodes | Features | Frequency |
|-----|---------|-------|----------|-----------|
| `metr-la` | METR-LA traffic speed | 207 | Speed | 5 min |
| `pems-bay` | PEMS-BAY traffic speed | 325 | Speed | 5 min |
| `pems04` | PEMS04 traffic | 307 | Flow, Occupancy, Speed | 5 min |
| `pems08` | PEMS08 traffic | 170 | Flow, Occupancy, Speed | 5 min |
| `beijing-air` | Beijing Air Quality | 36 | PM2.5 | 1 hour |
| `elergone` | Electricity consumption | 370 | Power | 15 min |
| `nyiso` | NYISO Integrated Load | 11 | Load | 1 hour |
| `nyc-covid` | NYT COVID-19 counties | 3,000+ | Cases, Deaths | 1 day |
| `mel-peds` | Melbourne Pedestrian | 55 | Count | 1 hour |

## Pipeline

For each dataset, `BenchmarkRunner` executes:

1. **Prepare** — download and cache via `DataWorkspace`
2. **Window** — sliding windows → `(x, y)` arrays (config fixed per dataset)
3. **Split** — temporal train / val / test (config fixed per dataset)
4. **Tensors** — convert to float32 `torch.Tensor` (NaN preserved)
5. **Fit** — `model.fit(x_train, y_train, x_val, y_val, adj, config)`
6. **Predict** — `model.predict(x_test, adj, config)`
7. **Metrics** — MAE, RMSE, MAPE in original units; NaN positions excluded

## Dependencies

- Python >= 3.10
- pandas, numpy
- torch (for `BenchmarkRunner`)
- gdown (optional, for PEMS-BAY / METR-LA)
