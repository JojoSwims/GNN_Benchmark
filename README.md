# GNN Benchmark

GNN Benchmark is an **end-to-end benchmark suite** for spatiotemporal forecasting models. It standardizes datasets, preprocessing, train/val/test splits, and metrics so model comparisons are fair and reproducible.

If you want to evaluate your own model in this repository, the benchmark supports these standard workflows:

- run on built-in benchmark datasets,
- plug in your own dataset (after transforming it to the benchmark IR),
- integrate your own model through a lightweight wrapper,
- compare metrics side-by-side with other supported models.

---

## Benchmark scope

This repository defines a shared protocol for evaluating forecasting models:

- consistent train/validation/test workflow,
- unified tensor shapes,
- standardized metric computation (MAE, RMSE, MAPE),
- reproducible, model-vs-model comparisons.

Use it when you want comparable results across GNN (or non-GNN) time-series forecasting models under the same pipeline.

---

## Installation

```bash
git clone https://github.com/JojoSwims/GNN_Benchmark.git
cd GNN_Benchmark

# Either: install into the current environment
./install.sh

# Or: create ./venv and install into it
./install.sh --venv
```

`install.sh` upgrades `pip`, installs everything in `requirements.txt`,
then verifies that each required package imports cleanly. It exits
non-zero on any failure so it works in CI. If you prefer to manage your
environment yourself, `pip install -r requirements.txt` is equivalent.

### Requirements

- Python 3.10+
- `numpy`, `pandas`, `scipy` — core data and numerics
- `torch` — required by every model wrapper and the runner
- `torchdiffeq` — ODE solver used by the MTGODE model
- `gdown` — Google Drive downloads (METR-LA, PEMS-BAY, EU-Load, LamaH-CE,
  NOAA, Divvy)
- `pyarrow` — parquet reader (NOAA, Divvy)
- `tables` (PyTables) — HDF5 reader (METR-LA, PEMS-BAY)

## Quick start: run a baseline in a few lines

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

This verifies your setup and runs the full pipeline once.

---

## What you can run out of the box

### Built-in datasets

The benchmark includes multiple preconfigured datasets (traffic, air quality, energy/load, mobility, etc.), each with:

- a dataset key,
- download/prep logic,
- benchmark windowing/splitting configuration,
- optional graph structure (`adj`) when available.

Common dataset keys include:

- `metr-la`
- `pems-bay`
- `pems04`
- `pems08`
- `beijing-air`, `beijing-air-cluster1`, `beijing-air-cluster2`
- `elergone`
- `eu-load`
- `nyc-covid`
- `lamah-ce`
- `noaa-buoy`
- `divvy-bikeshare-static`

> Tip: start with one dataset, then add more datasets after your first run completes.

### Built-in model wrappers

The repository includes wrappers/examples for multiple published models and baselines, so you can run them directly and compare results against your own model.

---

## Bring your own model: implement one wrapper

To benchmark your model, implement the benchmark model contract by subclassing
`BenchmarkModel` and pass the instance to `BenchmarkRunner.run`. Your wrapper
must define `name`, `fit`, and `predict`.

### Step 1: implement the wrapper

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gnn_benchmark.models import BenchmarkModel, TrainingHistory


class MyModel(BenchmarkModel):
    @property
    def name(self) -> str:
        return "MyModel"

    def fit(self, x_train, y_train, x_val, y_val, adj, config):
        # Tensors arrive unbatched. Build whatever DataLoader you want.
        # NaN = missing -- mask in your loss instead of zero-filling.
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=getattr(config, "batch_size", 64),
            shuffle=True,
        )
        net = build_my_network(x_train.shape, adj)        # your code
        opt = torch.optim.Adam(net.parameters(), lr=config.lr)

        train_loss, val_loss = [], []
        for _ in range(config.max_epochs):
            ...                                             # train one epoch
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)

        self._net = net
        return TrainingHistory(train_loss=train_loss, val_loss=val_loss)

    def predict(self, x_test, adj, config) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            y_pred = self._net(x_test, adj)               # your code
        # Must be returned in *original (unnormalised)* units, shape
        # [S, H, N, D_out], dtype float32 / float64.
        return y_pred.cpu().numpy()
```

### Step 2: run it

```python
from gnn_benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    workspace_dir="./benchmark_workspace",
    datasets=["metr-la", "pems-bay", "beijing-air"],
)
result = runner.run(MyModel(), config=MyModelConfig(lr=1e-3, max_epochs=50))
print(result.summary())
```

### Step 3 (optional): hyperparameter tune it

If your config is a dataclass, the bundled tuner will random-search over any
fields you list:

```python
from gnn_benchmark.tuning import HyperparameterTuner, LogUniform, Categorical

tuner = HyperparameterTuner(
    model_factory=lambda: MyModel(),
    base_config=MyModelConfig(max_epochs=50, batch_size=64),
    dataset_key="metr-la",
    workspace_dir="./benchmark_workspace",
    search_space={
        "lr":      LogUniform(1e-4, 5e-3),
        "dropout": Categorical([0.0, 0.1, 0.3]),
    },
    strategy="random", n_trials=18, seed=0,
)
print(tuner.run().summary())
```

### Tensor contract

- `x_*` tensors: shape `[S, L, N, D_in]` `float32` `torch.Tensor` (NaN = missing)
- `y_*` tensors: shape `[S, H, N, D_out]` `float32` `torch.Tensor` (NaN = missing)
- `adj`: `np.ndarray` of shape `[N, N]`, or `None` when the dataset has no graph
- predictions: `np.ndarray` of shape `[S, H, N, D_out]` in **original units**
- batching: your wrapper builds its own `DataLoader`; the harness does not impose one
- normalisation: your wrapper handles it; the harness passes raw values

---

## Bring your own dataset: implement one loader

A custom dataset is one subclass of `DatasetLoader`. The loader handles
download/parsing once; the harness handles windowing, splitting, tensorisation,
and metrics.

### Step 1: implement the loader

`download_and_convert` returns two DataFrames in the benchmark's intermediate
representation (IR):

- `series_df` — long-format time series with columns `[ts, node_id, <feature_1>, <feature_2>, ...]`. `ts` is a `pd.Timestamp` on a regular grid; missing values are `NaN`.
- `edges_df` — `[src, dst, cost]` rows for the static adjacency, or `None` if the dataset has no graph.

```python
from dataclasses import dataclass
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo, WindowConfig
from gnn_benchmark.datasets.base import DatasetLoader


@dataclass
class MyLoader(DatasetLoader):
    data_path: str = "./my_data.csv"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="my-dataset",
            url="https://example.org/my-dataset",
            frequency="1h",                              # pandas freq string
            node_order=[],                               # filled in download_and_convert
            feature_columns=["load"],
            units={"load": "MW"},
            window_config=WindowConfig(
                input_length=12, horizon=12,
                target_columns=["load"],
            ),
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        raw = pd.read_csv(self.data_path, parse_dates=["ts"])
        # Long format: one row per (timestamp, node_id, feature_value).
        series_df = raw[["ts", "node_id", "load"]].copy()
        series_df["node_id"] = series_df["node_id"].astype(str)

        # Optional static graph. Use None when no graph exists.
        edges_df = pd.DataFrame(
            [("A", "B", 1.0), ("B", "C", 1.0)],
            columns=["src", "dst", "cost"],
        )
        return series_df, edges_df
```

### Step 2: register and run

`DATASET_REGISTRY` is a plain dict, so a custom loader is one assignment away
from being usable through `BenchmarkRunner`:

```python
from gnn_benchmark import BenchmarkRunner
from gnn_benchmark.benchmark import DATASET_REGISTRY
from gnn_benchmark.models import LastValueModel

DATASET_REGISTRY["my-dataset"] = lambda: MyLoader(data_path="./my_data.csv")

runner = BenchmarkRunner(datasets=["my-dataset"])
print(runner.run(LastValueModel()).summary())
```

You can also skip the registry and just use the loader directly through
`DataWorkspace.prepare`, which is what the runner does internally:

```python
from gnn_benchmark import DataWorkspace
ir = MyLoader(data_path="./my_data.csv").prepare(DataWorkspace("./benchmark_workspace"))
```

### Practical checklist for custom datasets

- define a clear timestamp grid and `frequency`,
- represent missing data with `NaN` (not zero-filling),
- return raw units; the harness computes metrics in those units,
- provide an adjacency only when your domain has one — otherwise `edges_df=None`,
- verify shape and `node_order` consistency before running long training jobs.

---

## How comparison runs work

For each selected dataset, the runner executes:

1. dataset prepare/load,
2. sliding-window generation,
3. temporal train/val/test split,
4. tensor conversion,
5. model fit,
6. model predict,
7. metric evaluation on the held-out test set.

Because this sequence is shared across models, metrics are produced under the same evaluation flow.

---

## Reproducing the paper experiments

Every number reported in the paper comes from one of three example
directories. Each script is self-contained and writes to
`./benchmark_workspace/`, which is gitignored.

| To reproduce | Run |
|---|---|
| **Baselines** (LastValue and MLPMultivariate, no tuning) | `python examples_baseline/<dataset>_<lastvar\|mlp>_example.py` |
| **Main results** (every model × dataset, fair-protocol random search) | `python examples_new/<dataset>_<model>_example.py` |
| **Ablations** (no-graph and no-adaptive variants on the winning configs) | `python ablation_examples/<dataset>_<model>_ablation.py` |

For example, to reproduce GWN on EU-Load:

```bash
python examples_new/eu_load_gwn_example.py
```

The fair-tuning protocol used by every script in `examples_new/` is documented
in `examples_new/_shared.py`: identical `n_trials=18` random search,
`seed=0`, dataset-specific training schedule (batch size, max epochs,
LR milestones), and per-model search-space factory from
`gnn_benchmark.tuning.spaces`. Each script prints the per-trial summary,
the total tuning compute, and the final test-set evaluation with the
winning config.

To reproduce the entire paper end-to-end:

```bash
for f in examples_baseline/*.py examples_new/*.py ablation_examples/*.py; do
    python "$f"
done
```

(Reasonable runtimes assume a single GPU; LamaH-CE and NYC COVID are the
slowest; STAEFormer on NYC COVID is the most memory-intensive.)

## Recommended workflow for new users

1. Run `LastValueModel` on one built-in dataset to validate environment.
2. Run one or two existing wrapped models as reference baselines.
3. Add your model wrapper and confirm interface compatibility.
4. Benchmark your model on the same datasets and settings.
5. (Optional) add your own dataset via IR transform and repeat comparisons.

This order helps isolate setup issues (environment → data → model).

---

## Repository pointers

- `README.md` — high-level usage and onboarding (this file)
- `examples_baseline/` — simple starter usage
- `examples_new/` — model × dataset tuning/comparison examples
- `ablation_examples/` — graph/adaptive-edge ablation runs

---

## Common pitfalls

- Returning normalized predictions instead of original units
- Treating missing values as zeros instead of `NaN`
- Shape mismatches between your wrapper and benchmark contract
- Comparing models trained with different splits/settings outside the benchmark flow

These are common sources of invalid or inconsistent comparisons.

---

## In short

For benchmarking forecasting models, this repository provides a common framework:

- **use default datasets immediately**,
- **plug in custom data through an IR transform**,
- **integrate your model by implementing one wrapper**,
- **run competing models under the same protocol for fair comparison**.

