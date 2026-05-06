# Divvy Bikeshare Dataset — Implementation Notes

Reference doc for the Divvy Chicago bikeshare loader. Covers the loader's
parameterization and outputs, plus the framework-level changes that
were needed to support time-varying edges.

---

## 1. Data source

- **Upstream:** Divvy monthly trip CSVs (one zip per month) from
  `https://divvy-tripdata.s3.amazonaws.com/`.
- **Combined artifact:** a single parquet bundling many months, produced
  offline by `scripts/combine_divvy_tripdata.py` and uploaded once to
  Google Drive. ID = `1r2rfVNDgjcJiJO3lGOCK81AF7cDjHAKZ`. Current
  coverage: 2024-03 through 2026-03 (25 months, ~11.7 M trips,
  ~402 MB).
- **Download path** (when `data_path=None`): resolved in this order by
  [`DivvyBikeshareLoader._resolve_data_path`](datasets/divvy_bikeshare.py):
  1. Repo-local `datasets/divvy_tripdata_combined.parquet` if present
     (developer convenience — no re-download).
  2. Cache at `~/.cache/gnn_benchmark/divvy_bikeshare/divvy_tripdata_combined.parquet`.
  3. `gdown.download(id=GDRIVE_ID, ...)` → cache.
- **`gdown`** is a soft dependency (`try: import gdown`). `ImportError`
  with a helpful message if it's needed but missing. Already installed
  in this environment.

---

## 2. `DivvyBikeshareLoader` parameters

All kwargs are optional. File:
[`datasets/divvy_bikeshare.py`](datasets/divvy_bikeshare.py).

| kwarg | default | purpose |
|---|---|---|
| `data_path` | `None` | Source file. Accepts `.parquet` or `.csv`. `None` → GDrive download + cache (see §1). |
| `resolution` | `"1h"` | Pandas offset alias for the aggregation window. `"1h"` / `"6h"` / `"1D"`. Applies to both node series and dynamic-edge snapshots. |
| `edges_mode` | `"both"` | `"static"` (haversine only), `"dynamic"` (hourly trip-count snapshots only), `"both"` (both populated). Name suffix `_s` / `_d` / `_sd`. |
| `bbox` | `None` | `(lat_min, lat_max, lng_min, lng_max)`. If given, only stations inside the box are kept, and every trip with either endpoint outside the kept set is dropped — so `departures` / `arrivals` for kept stations aren't polluted by trips crossing in. |
| `top_k_stations` | `None` | Further trim the kept station set to the top K most-active stations (by total departures+arrivals). Applied after `bbox`. |
| `static_edge_cutoff_m` | `None` | Static haversine graph keeps only pairs within this many meters. `None` = fully connected. Big disk/IO win; meaningful for graph structure even when dense-matmul models still multiply by zeros. |

**Cache-key isolation:** `info.name` encodes `bbox`, `top_k`, cutoff, and
resolution. Switching any of these rebuilds instead of silently reusing
a stale cached IR.

---

## 3. Cleaning rules

Applied in order by `_read_and_clean`, drop counts logged per rule:

1. Missing `start_station_id` / `end_station_id` (dominant drop — ~31%
   of raw rows; Divvy anonymizes some trips).
2. Missing any of `start_lat` / `start_lng` / `end_lat` / `end_lng`.
3. `ended_at <= started_at` (corrupt / zero-duration).
4. Trip duration > 24 h (lost bikes / data errors).
5. Coords outside the Chicago bounding box (wide:
   lat ∈ [41.5, 42.2], lng ∈ [−88.2, −87.4]).
6. **(Optional)** `bbox` / `top_k_stations` subset filter. Determines
   the kept station set, then drops every trip with either endpoint
   outside the set.
7. Same-station round-trips (self-loops).
8. Duplicate `ride_id` (safety net; combine script already dedupes).

---

## 4. Outputs (IR contents)

### `ir.series` — `[ts, node_id, departures, arrivals, member_ratio]`

Long-format pandas DataFrame, densified to the full `(ts × node)` grid.
Zero-fill for counts (bikeshare zeros are real, not missing); NaN for
`member_ratio` where `departures == 0` (undefined).

- `departures` — trips starting at the station in window `[ts, ts+resolution)` (bucketed by `started_at`).
- `arrivals` — trips ending at the station in window (bucketed by `ended_at`).
- `member_ratio` — fraction of that station's departures made by annual
  members (the rest are casual riders).

**Target convention:** `window_config.target_columns=["departures"]`, and
because all four model adapters' `_FeatureScaler.inverse_transform`
slices `self.std[:D_out]`, the **target column must remain at index 0**
of `feature_columns`. Current order already satisfies this.

### `ir.edges` — `[src, dst, cost]` (if `edges_mode` in `{"static", "both"}`)

Symmetric haversine distance graph. Fully connected by default;
sparsified if `static_edge_cutoff_m` is set. Diagonal excluded. All
node-id references are in `metadata.node_order`.

### `ir.dynamic_edges` — `[ts, src, dst, cost]` (if `edges_mode` in `{"dynamic", "both"}`)

Hourly directed trip-count snapshots, **bucketed by `ended_at`**. For
each `(ts, src, dst)`, `cost` is the number of trips that *ended* in
the window `[ts, ts+resolution)` originating at `src` and finishing at
`dst`. Sparse: only rows with `cost ≥ 1` are emitted.

**Why `ended_at` bucketing** — a trip's destination is only observable
at trip end. Bucketing by `started_at` would assign an edge to a window
where the destination was still unknown, retroactively encoding future
arrival information. Under `ended_at`, every trip in `adj[t]` is fully
observed by `t+1`, matching the node-feature convention.

**Invariant (unit-tested):** for every `(ts, dst=B)`,
`sum_src ir.dynamic_edges.cost[ts, :, B] == ir.series.arrivals[ts, B]`
exactly.

### `ir.metadata`

Standard `IRMetadata` — `name`, `frequency`, `node_order`,
`feature_columns`, `units`, `source_url`. The `name` field carries the
cache-isolation tags (bbox / top-K / cutoff / resolution / edges-mode).

---

## 5. Framework-level changes

Three files changed outside the loader to support time-varying edges
and to register the dataset.

### `core/intermediate.py`

- **New optional attribute** `dynamic_edges: pd.DataFrame | None` on
  `IntermediateRepresentation`. Sparse snapshot DataFrame with
  `[ts, src, dst, cost]`. Defaults to `None` so every existing loader
  keeps working unchanged.
- **Snapshot accessor:**
  ```python
  def get_dynamic_adjacency_snapshot(
      self, ts: pd.Timestamp, default_diagonal: float = 0.0
  ) -> np.ndarray
  ```
  Returns an `(N, N)` adjacency for the requested timestamp, built on
  demand — not eagerly materialized, to avoid `(T, N, N)` tensors that
  balloon to 100s of GB at our scale.
- **Persistence:** `save()` writes `dynamic_edges.csv` when the
  attribute is set. `load()` reads it back, re-parses `ts` as
  `datetime64[ns]`, coerces `src` / `dst` to `str`. Parquet is the
  long-term target (dtype-preserving, ~10× smaller); CSV is the
  current fallback since `pyarrow` wasn't in `requirements.txt`.

### `core/workspace.py`

- **One getattr hook** in `prepare()`:
  ```python
  dynamic_edges_df = None
  build_dyn = getattr(loader, "build_dynamic_edges", None)
  if callable(build_dyn):
      dynamic_edges_df = build_dyn()
  ```
  The `DatasetLoader` base class is **unchanged** — loaders that don't
  define `build_dynamic_edges` simply don't hit this branch, so every
  existing loader is backward-compatible. The Divvy loader opts in by
  defining the method.

### `benchmark.py`

- **New import** of `DivvyBikeshareLoader`.
- **New `DATASET_REGISTRY` entry** — a lambda instead of a bare class,
  so the registry can bake in the chosen kwargs:
  ```python
  "divvy-bikeshare-static": lambda: DivvyBikeshareLoader(
      edges_mode="static",
      bbox=(41.87, 41.945, -87.67, -87.605),   # Loop + NNS + Lincoln Park
      static_edge_cutoff_m=2000,
  ),
  ```
  Central source of truth for all model-run example files — editing
  the lambda propagates to all four `divvy_bikeshare_{model}_example.py`
  scripts without per-file changes.

### `datasets/__init__.py`

- Export `DivvyBikeshareLoader` in `__all__` and document it alongside
  the other loaders in the module docstring.

---

## 6. Typical dataset shape

With the default registry config (bbox + 2 km cutoff):

| quantity | value |
|---|---|
| N (stations) | ~515 |
| T (hourly steps, 2024-03 → 2026-03) | 18,264 |
| C (features) | 3 |
| Static edges (2 km cutoff) | ~88,000 (vs ~265,000 fully connected) |
| Trips after cleaning | ~4.09 M |
| D2STGNN memory/layer @ batch=32 | ~0.34 GB (fits comfortably) |
