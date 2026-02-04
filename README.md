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
result = exporter.export_from_workspace(
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

### Loading Datasets

```python
from gnn_benchmark.datasets import (
    BeijingAirLoader,   # subdivision: "beijing", "cluster1", "cluster2", "all"
    PEMSBayLoader,      # Requires gdown
    MetroLALoader,      # Requires gdown
    PEMS04Loader,
    PEMS08Loader,
    ElergoneLoader,
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
from gnn_benchmark.exporters import WindowConfig, SplitConfig

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

# Export
result = exporter.export(ir, output_dir, window_config, split_config)
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
from gnn_benchmark.exporters import ModelExporter, ExportResult

class MyExporter(ModelExporter):
    @property
    def name(self) -> str:
        return "my_model"

    def export(self, ir, output_dir, window_config, split_config):
        # Use helper methods
        data = ir.to_tensor()
        x, y = self._create_sliding_windows(data, ...)
        train_x, val_x, test_x = self._split_by_time(x, ...)

        # Save in your format
        ...

        return ExportResult(files={"data": path}, ...)
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

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{gnn_benchmark,
  title = {GNN Benchmark: A Toolkit for Benchmarking Graph Neural Networks on Spatiotemporal Data},
  year = {2024},
  url = {https://github.com/JojoSwims/GNN_Benchmark}
}
```
