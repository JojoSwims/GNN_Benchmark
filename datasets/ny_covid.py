import csv
import json
import urllib.request
from itertools import combinations
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import pandas as pd


# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("nyt_covid_data")
DATA_DIR.mkdir(exist_ok=True)

YEARS = [2020, 2021, 2022, 2023]

NYT_RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-{year}.csv"
FIPS_MAP_RAW_URL = "https://raw.githubusercontent.com/josh-byster/fips_lat_long/master/fips_map.json"

OUT_MEASUREMENTS = DATA_DIR / "covid_measurements_2020_2023.csv"
OUT_DISTANCES = DATA_DIR / "fips_distances.csv"


# ----------------------------
# Distance function (meters)
# ----------------------------
def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Returns distance in meters.
    """
    R = 6371008.8  # Mean Earth radius in meters

    phi1, lambda1, phi2, lambda2 = map(radians, (lat1, lon1, lat2, lon2))

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


# ----------------------------
# Download helper
# ----------------------------
def download_file(url: str, out_path: Path) -> None:
    print(f"[DOWNLOAD] {url}")
    with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
        f.write(response.read())
    print(f"[SAVED]    {out_path}")


print("\n=== STEP 1/7: Download source files ===")
csv_paths = []
for year in YEARS:
    path = DATA_DIR / f"us-counties-{year}.csv"
    download_file(NYT_RAW_URL_TEMPLATE.format(year=year), path)
    csv_paths.append(path)

fips_map_path = DATA_DIR / "fips_map.json"
download_file(FIPS_MAP_RAW_URL, fips_map_path)


print("\n=== STEP 2/7: Load NYT county data ===")
frames = []
for p in csv_paths:
    print(f"[LOAD] {p.name}")
    df = pd.read_csv(p, usecols=["date", "county", "state", "fips", "cases", "deaths"])
    df["fips"] = pd.to_numeric(df["fips"], errors="coerce").astype("Int64")
    frames.append(df)

covid = pd.concat(frames, ignore_index=True)
covid["date"] = pd.to_datetime(covid["date"], format="%Y-%m-%d")
print(f"[INFO] Loaded {len(covid):,} rows from NYT files")

# NYC patch: assign synthetic FIPS
nyc_mask = (covid["county"] == "New York City") & (covid["state"] == "New York")
covid.loc[nyc_mask, "fips"] = 99999
print(f"[PATCH] Assigned synthetic FIPS 99999 to NYC rows: {int(nyc_mask.sum()):,} rows")


print("\n=== STEP 3/7: Load FIPS coordinates ===")
with open(fips_map_path, "r", encoding="utf-8") as f:
    raw_fips_map = json.load(f)

coords = pd.DataFrame(
    [
        {"fips": int(k), "lat": float(v["lat"]), "lon": float(v["long"])}
        for k, v in raw_fips_map.items()
    ]
).drop_duplicates(subset=["fips"], keep="first")

# Add NYC synthetic coordinate
coords = pd.concat(
    [coords, pd.DataFrame([{"fips": 99999, "lat": 40.7128, "lon": -74.0060}])],
    ignore_index=True
).drop_duplicates(subset=["fips"], keep="first")

coords["fips"] = coords["fips"].astype("int64")
print(f"[INFO] Loaded coordinates for {len(coords):,} FIPS (including NYC synthetic FIPS)")


print("\n=== STEP 4/7: Merge and keep geolocated rows ===")
merged_raw = covid.merge(coords, on="fips", how="left")

# Report what gets dropped (has FIPS but no coordinates)
unmapped_fips = (
    merged_raw[
        merged_raw["fips"].notna() & (merged_raw["lat"].isna() | merged_raw["lon"].isna())
    ][["county", "state", "fips"]]
    .drop_duplicates()
    .sort_values(["state", "county"])
    .reset_index(drop=True)
)

missing_fips_groups = (
    merged_raw[merged_raw["fips"].isna()][["county", "state"]]
    .drop_duplicates()
    .sort_values(["state", "county"])
    .reset_index(drop=True)
)

print(f"[INFO] Unique missing-FIPS groups: {len(missing_fips_groups)}")
if not missing_fips_groups.empty:
    print(missing_fips_groups.to_string(index=False))

print(f"[INFO] Unique FIPS with no coordinates (will be dropped): {len(unmapped_fips)}")
if not unmapped_fips.empty:
    print(unmapped_fips.to_string(index=False))

merged = merged_raw[merged_raw["lat"].notna() & merged_raw["lon"].notna()].copy()
merged = merged[["date", "fips", "cases", "deaths"]].copy()
merged["fips"] = merged["fips"].astype("int64")

print(f"[INFO] Geolocated rows kept: {len(merged):,}")
print(f"[INFO] Rows dropped: {len(merged_raw) - len(merged):,}")


print("\n=== STEP 5/7: Build complete county-date panel (fill missing timestamps) ===")
all_dates = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
all_fips = sorted(merged["fips"].unique())

print(f"[INFO] Date range: {all_dates.min().strftime('%Y-%m-%d')} → {all_dates.max().strftime('%Y-%m-%d')} ({len(all_dates)} days)")
print(f"[INFO] Unique FIPS in final dataset: {len(all_fips):,}")

full_index = pd.MultiIndex.from_product([all_dates, all_fips], names=["date", "fips"])
panel = merged.set_index(["date", "fips"]).reindex(full_index).reset_index()

# Nullable integers -> blank cells in CSV
panel["cases"] = pd.array(panel["cases"], dtype="Int64")
panel["deaths"] = pd.array(panel["deaths"], dtype="Int64")

panel_out = panel.rename(columns={"date": "timestamp"}).copy()
panel_out["timestamp"] = panel_out["timestamp"].dt.strftime("%Y-%m-%d")
panel_out = panel_out[["timestamp", "fips", "cases", "deaths"]]

print(f"[INFO] Completed panel rows: {len(panel_out):,}")
print("[SAVE] Writing consolidated measurements file...")
panel_out.to_csv(OUT_MEASUREMENTS, index=False)
print(f"[SAVED] {OUT_MEASUREMENTS}")


print("\n=== STEP 6/7: Build symmetric FIPS distance file ===")
county_coords = coords[coords["fips"].isin(all_fips)].copy().sort_values("fips").reset_index(drop=True)

n = len(county_coords)
num_undirected_pairs = n * (n - 1) // 2
num_directed_edges = n * (n - 1)

print(f"[INFO] FIPS nodes: {n:,}")
print(f"[INFO] Undirected pairs to compute: {num_undirected_pairs:,}")
print(f"[INFO] Directed edges to write: {num_directed_edges:,}")
print("[SAVE] Writing distance edges (this may take a while)...")

progress_every = 250_000  # undirected pairs
pairs_done = 0

with open(OUT_DISTANCES, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["from", "to", "cost"])

    rows = county_coords.to_dict("records")
    for a, b in combinations(rows, 2):
        fa, fb = int(a["fips"]), int(b["fips"])
        dist = round(_haversine_distance(a["lat"], a["lon"], b["lat"], b["lon"]), 1)

        # symmetric directed edges
        writer.writerow([fa, fb, dist])
        writer.writerow([fb, fa, dist])

        pairs_done += 1
        if pairs_done % progress_every == 0:
            pct = 100 * pairs_done / num_undirected_pairs
            print(f"[PROGRESS] Distance pairs: {pairs_done:,}/{num_undirected_pairs:,} ({pct:.1f}%)")

print(f"[SAVED] {OUT_DISTANCES}")


print("\n=== STEP 7/7: Done ===")
print("Outputs:")
print(f" - Measurements: {OUT_MEASUREMENTS}")
print(f" - Distances:     {OUT_DISTANCES}")