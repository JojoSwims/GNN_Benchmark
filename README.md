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

pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- `numpy`, `pandas`, `scipy` — core data and numerics
- `torch` — required by every model wrapper and the runner
- `torchdiffeq` — ODE solver used by the MTGODE model
- `gdown` — Google Drive downloads (METR-LA, PEMS-BAY, EU-Load, LamaH-CE,
  NOAA, Divvy)
- `pyarrow` — parquet reader (NOAA, Divvy)
- `tables` (PyTables) — HDF5 reader (METR-LA, PEMS-BAY)

---

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

To benchmark your model, implement the benchmark model contract by subclassing `BenchmarkModel`.

Your wrapper must define:

- `name` (string for reporting)
- `fit(...)`
- `predict(...)`

Minimal structure:

```python
from gnn_benchmark.models import BenchmarkModel

class MyModel(BenchmarkModel):
    @property
    def name(self) -> str:
        return "MyModel"

    def fit(self, x_train, y_train, x_val, y_val, adj, config):
        # Train your model here
        return None

    def predict(self, x_test, adj, config):
        # Return predictions in original units
        ...
```

### Important interface expectations

- `x_*` tensors are shaped `[S, L, N, D_in]`
- `y_*` tensors are shaped `[S, H, N, D_out]`
- missing values are represented as `NaN`
- `adj` can be `None` when no graph is provided
- predictions must be returned in **original (unnormalized) units**
- batching/data loaders are handled inside your wrapper (the harness does not impose one)

If your wrapper satisfies this contract, your model can run in the same pipeline as other models.

---

## Bring your own dataset: transform it to benchmark IR

You can benchmark custom data as long as you map it into the benchmark’s intermediate representation (IR) expected by the pipeline.

At a high level, your transform should produce:

- a multivariate time-series signal aligned in time,
- node dimension `N` (or equivalent entities),
- input/output feature dimensions,
- optional adjacency matrix `adj` (`[N, N]`) or `None`.

Then the benchmark can apply the standard windowing, split, tensor conversion, and evaluation flow.

### Practical checklist for custom datasets

- define clear timestamp alignment and frequency,
- represent missing data with `NaN` (not zero-filling by default),
- keep raw units available for final metric computation,
- provide graph structure only if your domain has one,
- verify shape consistency before running long training jobs.

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

