INTERMEDIATE CSV FORMAT
================================
NOTE: I asked ChatGPT to summarize my messy notes, this is written by AI.

Scope
-----
Minimal, code‑light interchange for multivariate time series with one value per node per timestamp, plus an optional static graph.

Files
-----
Required
1) **series.csv** — long/tidy table

Optional
2) **edges.csv** — static graph (edge list)
3) **mask.csv** — observation mask (only if missing values are *present*)
4) **metadata.json** — tiny manifest (recommended)

`series.csv`
-----------
Columns: `ts, node_id, value`
- `ts`: ISO‑8601 string **without timezone** (e.g., `2023-05-01T12:30:00`).
- One row per (`ts`,`node_id`); the pair must be unique.
- `value` numeric. If missing values are *present*, allow empty/NaN and provide `mask.csv`.

`edges.csv` (optional)
----------------------
Columns: `src, dst[, weight]`
- All nodes must exist in `series.csv`.
- Undirected graphs: list one row per unordered pair (set `symmetric: true` in metadata).

`mask.csv` (optional; required if `missing == "imputed"`)
---------------------------------------------------------
Columns: `ts, node_id, is_observed` (0/1 or true/false)
- Keys must match (`ts`,`node_id`) in `series.csv`.

`metadata.json` (recommended)
----------------------------
Minimal fields:
```json
{
  "name": "Example",
  "units": "speed_kph",
  "freq": { "value": 5, "unit": "minute" },   // or null if irregular
  "missing": "none" | "imputed" | "present",
  "adjacency": {                                   // include only if edges.csv exists
    "unit": "km" | "s" | "prob" | "corr",
    "meaning": "distance" | "travel_time" | "connectivity" | "correlation",
    "directed": true,
    "symmetric": false,
    "allow_self_loops": false
  }
}
```

Validation (lightweight)
------------------------
- (`ts`,`node_id`) unique in `series.csv`.
- `value` numeric; `ts` parseable as ISO‑8601.
- Missing policy:
  - `none` → no NaN in `series.csv`, no `mask.csv`.
  - `imputed` → numeric only; imputation handled upstream; provide `mask.csv`.
  - `present` → `series.csv` may contain NaN
- Graph nodes in `edges.csv` must exist in `series.csv`.
