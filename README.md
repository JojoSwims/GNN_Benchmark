# GNN Benchmark

A modular Python toolkit for benchmarking Graph Neural Networks (GNNs) on spatiotemporal time series data.

## Overview

GNN Benchmark provides a clean, unified pipeline for:

- **Loading datasets** from external sources (traffic, air quality, energy)
- **Applying transforms** to intermediate representations (imputation, normalization, temporal features)
- **Exporting data** in model-specific formats (STAEformer, GraphWaveNet, ASTGCN, etc.)
- **Running baseline models** for comparison (Persistence, Historical Average, SARIMA)

## Installation

```bash
# Clone the repository
git clone https://github.com/JojoSwims/GNN_Benchmark.git
cd GNN_Benchmark

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for specific features
pip install gdown      # For downloading from Google Drive (PEMS-BAY, METR-LA)
pip install statsmodels  # For Kalman imputation and SARIMA baseline
pip install h5py       # For HDF5 export format (MTGNN)
```

## Quick Start

```python
from gnn_benchmark import DataWorkspace
from gnn_benchmark.datasets import BeijingAirLoader
from gnn_benchmark.transforms import FillZeros, AddTimeFeatures, ZScoreNormalize
from gnn_benchmark.exporters import STAEformerExporter, WindowConfig

# 1. Create a workspace
workspace = DataWorkspace("./my_workspace")

# 2. Load a dataset
loader = BeijingAirLoader(subdivision="beijing")
ir = loader.prepare(workspace)

# 3. Apply transforms
ir.apply(FillZeros())
ir.apply(AddTimeFeatures(time_of_day=True, day_of_week=True))
ir.apply(ZScoreNormalize(train_end="2015-01-01"))

# 4. Export for a specific model
exporter = STAEformerExporter()
result = exporter.export(
    ir,
    window_config=WindowConfig(input_length=12, horizon=12)
)

print(f"Exported files: {result.files}")
```

## Architecture

```
Dataset Loader  →  Intermediate Representation  →  Transforms  →  Exporter  →  Model Format
                            ↓
                       DataWorkspace
                     (file management)
```

### Workspace Structure

```
workspace/
├── clean/           # Original converted data (immutable)
│   └── {dataset}/
│       ├── series.csv
│       ├── edges.csv
│       └── metadata.json
├── working/         # Transformed data (mutable)
│   └── {dataset}/
└── exports/         # Model-specific outputs
    └── {dataset}/
        ├── staeformer/
        ├── graph_wavenet/
        └── ...
```

## Core Concepts

### Intermediate Representation (IR)

The IR is the central data structure that unifies all datasets into a common format:

- **Series DataFrame**: Long format with columns `[ts, node_id, feature1, feature2, ...]`
- **Edges DataFrame** (optional): Graph structure with columns `[src, dst, cost]`
- **Mask DataFrame** (optional): Observation mask with columns `[ts, node_id, is_observed]`
- **Metadata**: Dataset info, transform history, normalization stats

```python
# Access IR properties
print(ir.shape)           # (T, N, C) - timestamps, nodes, channels
print(ir.nodes)           # List of node IDs
print(ir.timestamps)      # DatetimeIndex of timestamps
print(ir.feature_columns) # List of feature names

# Convert to tensor
tensor = ir.to_tensor()   # Returns (T, N, C) numpy array

# Get adjacency matrix
adj = ir.get_adjacency_matrix()  # Returns (N, N) numpy array
```

### Transforms

Transforms modify the IR in-place and record their actions in the transform history:

```python
from gnn_benchmark.transforms import (
    # Imputation
    FillZeros,        # Replace NaN with 0, create observation mask
    ForwardFill,      # Forward fill within each node's series
    KalmanImpute,     # Kalman filter with seasonal components

    # Temporal
    AddTimeFeatures,  # Add time_of_day, day_of_week, time_of_year
    FilterHours,      # Keep only specific hours
    FilterDateRange,  # Keep only specific date range
    Resample,         # Resample to different frequency

    # Normalization
    ZScoreNormalize,  # Z-score using training statistics
    MinMaxNormalize,  # Min-max to [0, 1] using training statistics
    Denormalize,      # Reverse normalization
)

# Apply transforms (chainable)
ir.apply(FillZeros()).apply(AddTimeFeatures())

# Save the transformation:
ir.save()

# Check transform history
print(ir.metadata.transform_history)
# ['FillZeros(columns=all)', 'AddTimeFeatures(tod, dow)']
```

## Available Datasets

| Dataset | Nodes | Features | Frequency | Source |
|---------|-------|----------|-----------|--------|
| Beijing Air Quality | 36-437 | PM2.5 | 1 hour | Microsoft Research |
| PEMS-BAY | 325 | Speed | 5 min | CalTrans |
| METR-LA | 207 | Speed | 5 min | CalTrans |
| PEMS04 | 307 | Flow, Occupancy, Speed | 5 min | CalTrans |
| PEMS08 | 170 | Flow, Occupancy, Speed | 5 min | CalTrans |
| Elergone | 370 | Power | 15 min | UCI ML Repository |
| NYISO Integrated Load | 11 | Load | 1 hour | NYISO |
| Melbourne Pedestrian Counts | 55 | Count | 1 hour | UCTB Urban Dataset |
| NYT COVID-19 (US Counties) | 3,000+ | Cases, Deaths | 1 day | The New York Times |

### Loading Datasets

```python
from gnn_benchmark.datasets import (
    BeijingAirLoader,   # subdivision: "beijing", "cluster1", "cluster2", "all"
    PEMSBayLoader,      # Requires gdown
    MetroLALoader,      # Requires gdown
    PEMS04Loader,
    PEMS08Loader,
    ElergoneLoader,
    NYISOLoader,
    MelPedsLoader,
    NYCovidLoader,
)

# Example: Load Beijing Air with different subdivisions
loader = BeijingAirLoader(subdivision="beijing")  # 36 nodes
loader = BeijingAirLoader(subdivision="cluster1") # 284 nodes
loader = BeijingAirLoader(subdivision="all")      # 437 nodes
```

## Available Exporters

| Exporter | Output Format | Description |
|----------|--------------|-------------|
| STAEformerExporter | `data.npz`, `index.npz` | Full tensor + index arrays for runtime windowing |
| GraphWaveNetExporter | `train.npz`, `val.npz`, `test.npz`, `adj_mx.pkl` | Pre-windowed splits |
| ASTGCNExporter | `data.npz`, split files, `sensor_ids.txt` | Full tensor + windowed splits |
| MTGNNExporter | `data.h5`, `adj_mx.pkl`, split files | HDF5 format |
| GTSExporter | Similar to GraphWaveNet | For GTS model |
| D2STGCNExporter | Similar to GraphWaveNet | For D2STGCN model |

### Export Configuration

```python
from gnn_benchmark.exporters import STAEformerExporter, WindowConfig, SplitConfig

# Configure sliding windows
window_config = WindowConfig(
    input_length=12,      # L: past timesteps
    horizon=12,           # H: future timesteps to predict
    y_start=1,            # Gap between input end and target start
    input_columns=None,   # None = all features
    target_columns=None,  # None = same as input
)

# Configure train/val/test split
split_config = SplitConfig(
    train_ratio=0.7,
    val_ratio=0.1,
    # test_ratio = 0.2 (implicit)
)

# Export the IR (must be linked to workspace)
exporter = STAEformerExporter()
result = exporter.export(ir, window_config, split_config)
```

## Baseline Models

```python
from gnn_benchmark.baselines import Persistence, HistoricalAverage, SARIMA

# Evaluate persistence baseline
baseline = Persistence(fill_method="forward")
result = baseline.evaluate(
    ir,
    train_end=pd.Timestamp("2015-01-01"),
    val_end=pd.Timestamp("2015-03-01"),
    horizons=[1, 3, 6, 12],
)

print(f"Overall MAE: {result.metrics['mae']:.4f}")
print(f"Per-horizon metrics: {result.per_horizon_metrics}")
```

| Model | Description |
|-------|-------------|
| Persistence | Predicts last observed value: y_hat(t+h) = y(t) |
| HistoricalAverage | Predicts average from same period in history |
| SARIMA | Seasonal ARIMA with optional Kalman imputation |

## Model Submission Contract

To benchmark a custom GNN model, subclass `BenchmarkModel` and implement
`fit` and `predict`.

```python
from gnn_benchmark.models import BenchmarkModel, TrainingHistory

class MyGNN(BenchmarkModel):
    name = "MyGNN"

    def fit(self, train_loader, val_loader, adj, config):
        """
        Train the model.

        Args:
            train_loader: DataLoader yielding (x_batch, y_batch) with
                          x_batch [B, seq_in_len, N, D_in] float32
                          y_batch [B, seq_out_len, N, D_out] float32
            val_loader:   Same format, shuffle=False.
            adj:          Adjacency matrix [N, N] float32, or None if the
                          dataset has no edge information.
            config:       Any config object you choose (dataclass, dict, …).
        """
        # ... train your model ...
        return TrainingHistory(train_loss=[...], val_loss=[...])  # or None

    def predict(self, test_loader, adj, config):
        """
        Generate predictions.

        Returns:
            np.ndarray of shape [num_test_samples, seq_out_len, N, D_out]
            in the same normalised space as the training targets.
        """
        # ... run inference ...
        return y_pred

    def get_config(self) -> dict:
        """Optional: expose hyperparameters for result logging."""
        return {"hidden_dim": 64, "layers": 3}
```

### Contract rules

| Rule | Detail |
|------|--------|
| **DataLoader format** | Loaders yield `(x_batch, y_batch)` float32 tensors |
| **Test order** | `test_loader` is always `shuffle=False` — row order matches ground truth |
| **Adjacency** | `adj` is `None` for datasets with no graph; raise `ValueError` if you require it |
| **Normalisation** | Inputs and targets are Z-score normalised; return `y_pred` in the same space |
| **Config** | Fully submitter-controlled — pass whatever your model needs |

### Building DataLoaders manually

```python
from gnn_benchmark.exporters import create_sliding_windows, split_by_time, make_dataloaders
from gnn_benchmark.exporters import WindowConfig, SplitConfig

data = ir.to_tensor()                            # (T, N, C)
x, y = create_sliding_windows(data, input_length=12, horizon=12)
x_train, x_val, x_test = split_by_time(x, 0.7, 0.1)
y_train, y_val, y_test = split_by_time(y, 0.7, 0.1)

train_loader, val_loader, test_loader = make_dataloaders(
    x_train, y_train, x_val, y_val, x_test, y_test, batch_size=32
)
```

---

## Running the Benchmark

`BenchmarkRunner` drives the full pipeline — download, preprocess, window,
fit, predict, and score — for every dataset you specify.

```python
from benchmark import BenchmarkRunner
from gnn_benchmark.exporters import WindowConfig, SplitConfig

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["metr-la", "pems-bay", "pems04"],  # None = all datasets
    window_config=WindowConfig(input_length=12, horizon=12),
    split_config=SplitConfig(train_ratio=0.7, val_ratio=0.1),
    batch_size=32,
)

result = runner.run(MyGNN(), config=my_config)
print(result.summary())
```

Sample output:

```
Model: MyGNN
────────────────────────────────────────────────────────
Dataset              MAE      RMSE     MAPE
────────────────────────────────────────────────────────
metr-la           0.4123   0.6814   9.21%
pems-bay          0.3187   0.5241   7.83%
pems04            0.5214   0.8427  11.40%
────────────────────────────────────────────────────────
Mean              0.4175   0.6827   9.48%
────────────────────────────────────────────────────────
(metrics in normalised space)
```

### Available datasets

| Key | Dataset |
|-----|---------|
| `metr-la` | METR-LA (requires `gdown`) |
| `pems-bay` | PEMS-BAY (requires `gdown`) |
| `pems04` | PEMS04 |
| `pems08` | PEMS08 |
| `beijing-air` | Beijing Air Quality |
| `elergone` | Elergone (no graph) |
| `nyiso` | NYISO Integrated Load (no graph) |
| `nyc-covid` | NYT COVID-19 US Counties |
| `mel-peds` | Melbourne Pedestrian Counts |

### Accessing detailed results

```python
# Per-dataset metrics
r = result.dataset_results["metr-la"]
print(r.mae, r.rmse, r.mape)

# Per-horizon breakdown
for step, metrics in r.per_horizon.items():
    print(f"  step {step:2d}: MAE={metrics['mae']:.4f}")

# Export to dict / JSON
import json
print(json.dumps(result.to_dict(), indent=2))
```

### Adding PyTorch

The `make_dataloaders` helper and the benchmark runner require PyTorch:

```bash
pip install torch
```

---

## Utility Functions

### Evaluation Metrics

```python
from gnn_benchmark.utils import mae, rmse, mape, mse, smape, r2_score

# All metrics support optional boolean mask
mae_value = mae(y_true, y_pred)
rmse_value = rmse(y_true, y_pred, mask=valid_mask)
```

### Graph Utilities

```python
from gnn_benchmark.utils import (
    adjacency_from_edges,      # Convert edges DataFrame to matrix
    normalize_adjacency,       # Symmetric normalization D^(-1/2) A D^(-1/2)
    random_walk_matrix,        # Random walk normalization D^(-1) A
    normalized_laplacian,      # I - D^(-1/2) A D^(-1/2)
    scaled_laplacian,          # For ChebNet: 2L/lambda_max - I
    exp_decay_adjacency,       # Exponential decay weights from distances
    to_sparse,                 # Convert to COO sparse format
)
```

### I/O Utilities

```python
from gnn_benchmark.utils import (
    save_pickle, load_pickle,
    save_npz, load_npz,
    save_json, load_json,
    ensure_dir,
    list_files,
)
```

## API Reference

### Core Classes

#### `DataWorkspace`

Central manager for data files and operations.

```python
workspace = DataWorkspace("./my_workspace")

# Check status
workspace.is_downloaded("beijing_air")
workspace.has_working("beijing_air")
workspace.list_datasets()
workspace.list_exports("beijing_air")

# Load data
ir = workspace.load("beijing_air")
ir = workspace.load("beijing_air", from_clean=True)

# Prepare dataset (download + convert)
ir = workspace.prepare(loader)

# Export to model format
result = workspace.export(exporter, ir, window_config, split_config)

# Save/clear
workspace.save_working(ir)
workspace.clear_working("beijing_air")
workspace.clear_exports("beijing_air", exporter_name="staeformer")
```

#### `IntermediateRepresentation`

Central data object wrapping time series and graph data.

```python
# Properties
ir.shape          # (T, N, C)
ir.nodes          # List[str]
ir.timestamps     # pd.DatetimeIndex
ir.feature_columns # List[str]

# Transform operations
ir.apply(transform)
ir.reset_transforms()

# Data conversion
ir.to_tensor(columns=["value"])
ir.get_adjacency_matrix()
ir.get_observation_mask()

# Splitting
train_end, val_end = ir.get_split_timestamps(train_ratio=0.7, val_ratio=0.1)

# Persistence
ir.save(path)
ir = IntermediateRepresentation.load(path)
```

### Creating Custom Components

#### Custom Dataset Loader

```python
from gnn_benchmark.datasets import DatasetLoader
from gnn_benchmark.core.types import DatasetInfo

class MyDatasetLoader(DatasetLoader):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="my_dataset",
            url="https://...",
            frequency="1H",
            node_order=["node1", "node2", ...],
            feature_columns=["value"],
            units={"value": "units"},
            description="My custom dataset",
        )

    def download_and_convert(self):
        # Download and process your data
        series_df = ...  # DataFrame with [ts, node_id, value, ...]
        edges_df = ...   # DataFrame with [src, dst, cost] or None
        return series_df, edges_df
```

#### Custom Transform

```python
from gnn_benchmark.transforms import Transform

class MyTransform(Transform):
    @property
    def description(self) -> str:
        return "MyTransform()"

    def __call__(self, ir):
        # Modify ir.series, ir.mask, ir.metadata in-place
        ir.series["new_column"] = ir.series["value"] * 2
        ir.metadata.feature_columns.append("new_column")
```

#### Custom Exporter

```python
from gnn_benchmark.exporters import ModelExporter, ExportResult, WindowConfig, SplitConfig
from pathlib import Path

class MyExporter(ModelExporter):
    @property
    def name(self) -> str:
        return "my_model"

    def export_to_directory(
        self,
        ir,
        output_dir: Path,
        window_config: WindowConfig,
        split_config: SplitConfig,
    ) -> ExportResult:
        # Use helper methods
        data = ir.to_tensor()
        x, y = self._create_sliding_windows(
            data,
            window_config.input_length,
            window_config.horizon,
        )
        train_x, val_x, test_x = self._split_by_time(
            x,
            split_config.train_ratio,
            split_config.val_ratio,
        )

        # Save in your format
        output_dir.mkdir(parents=True, exist_ok=True)
        ...

        return ExportResult(
            files={"data": output_dir / "data.npz"},
            window_config=window_config,
            split_config=split_config,
        )

# Usage: exporter.export(ir, window_config, split_config)
```

## Data Format Specifications

### Series DataFrame (Long Format)

```
ts (datetime)    node_id (str)    feature1 (float)    feature2 (float)
2024-01-01 00:00 A                10.5                100.0
2024-01-01 00:00 B                12.3                105.0
2024-01-01 01:00 A                11.2                102.0
...
```

### Edges DataFrame

```
src (str)    dst (str)    cost (float)
A            B            2.5
B            C            1.8
A            C            4.2
```

### Metadata JSON

```json
{
  "name": "dataset_name",
  "frequency": "1H",
  "node_order": ["A", "B", "C"],
  "feature_columns": ["value"],
  "units": {"value": "units"},
  "source_url": "https://...",
  "transform_history": ["FillZeros()", "ZScoreNormalize(...)"],
  "extra": {"zscore_means": {...}, "zscore_stds": {...}}
}
```

## Dependencies

### Required

- Python >= 3.10
- pandas
- numpy

### Optional

- `gdown`: For downloading PEMS-BAY and METR-LA from Google Drive
- `statsmodels`: For Kalman imputation and SARIMA baseline
- `h5py`: For HDF5 export format (MTGNN)

```
