"""Download and extract the LamaH-CE dataset from Zenodo.

Shared helper used by both LamaHCELoader and LamaHCEDynamicLoader so the
loaders "just work" (like BeijingAirLoader) without the caller having to
manually fetch a 1.5 GB tarball.

The tarball and the extracted directory are cached under
``~/.cache/gnn_benchmark/lamah_ce/`` by default so subsequent runs pay
only the IR cache cost inside the workspace.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

# Zenodo record: https://doi.org/10.5281/zenodo.4525244
# The daily-only variant (~1.5 GB) is enough for `resolution="daily"`; the
# combined daily+hourly archive (~15 GB) is required for hourly runs.
_ZENODO_TARBALLS = {
    "daily":  ("2_LamaH-CE_daily.tar.gz",
               "https://zenodo.org/records/4525244/files/2_LamaH-CE_daily.tar.gz"),
    "hourly": ("1_LamaH-CE_daily_hourly.tar.gz",
               "https://zenodo.org/records/4525244/files/1_LamaH-CE_daily_hourly.tar.gz"),
}


def default_cache_dir() -> Path:
    """Persistent cache shared across workspaces."""
    return Path.home() / ".cache" / "gnn_benchmark" / "lamah_ce"


def ensure_lamah_data_root(
    resolution: str,
    cache_dir: Path | None = None,
) -> Path:
    """Return a local LamaH-CE data_root, downloading + extracting if needed.

    Args:
        resolution: "daily" or "hourly". "hourly" requires the combined
            archive (daily is a strict subset of hourly in terms of files).
        cache_dir: Override the default cache location.

    Returns:
        Path to the extracted LamaH-CE root (the folder containing
        ``D_gauges/``, ``B_basins_intermediate_all/``, …).
    """
    if resolution not in _ZENODO_TARBALLS:
        raise ValueError(
            f"resolution must be 'daily' or 'hourly', got {resolution!r}"
        )
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    filename, url = _ZENODO_TARBALLS[resolution]
    tarball_path = cache_root / filename
    extract_dir = cache_root / f"extracted_{resolution}"
    marker = extract_dir / ".extracted_ok"

    if marker.exists():
        return _find_data_root(extract_dir)

    # Download (atomic: write to .part then rename)
    if not tarball_path.exists():
        print(f"[LamaH-CE] Downloading {url}")
        print(f"[LamaH-CE] This is a large file; first-run fetch takes a while.")
        tmp = tarball_path.with_suffix(tarball_path.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(tarball_path)

    # Extract
    print(f"[LamaH-CE] Extracting {tarball_path.name} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path) as tf:
        tf.extractall(extract_dir)
    marker.touch()
    return _find_data_root(extract_dir)


def _find_data_root(path: Path) -> Path:
    """Locate the LamaH-CE root inside an extracted directory.

    The tarball layout typically nests files under one extra directory
    (e.g. ``2_LamaH-CE_daily/``), so we search for the canonical
    ``D_gauges/`` marker and return its parent.
    """
    for p in path.rglob("D_gauges"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(
        f"Could not find D_gauges/ under {path}. "
        "The archive layout may have changed; delete the cache and retry."
    )
