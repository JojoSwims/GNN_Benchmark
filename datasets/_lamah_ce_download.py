"""Download and extract the LamaH-CE dataset from Google Drive.

Shared helper used by both LamaHCELoader and LamaHCEDynamicLoader so the
loaders "just work" (like BeijingAirLoader) without the caller having to
manually fetch a multi-GB tarball.

The tarball and the extracted directory are cached under
``~/.cache/gnn_benchmark/lamah_ce/`` by default so subsequent runs pay
only the IR cache cost inside the workspace.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

# Pre-mirrored LamaH-CE archive on Google Drive. Same daily layout as the
# Zenodo release (record 4525244) but hosted where we get a reliable fetch
# instead of intermittent 404s.
_LAMAH_GDRIVE_ID = "1-gMtrag7EtAqhuMB2sJfc85whd8iRLEt"
_LAMAH_TARBALL_NAME = "lamah_ce.tar.gz"


def default_cache_dir() -> Path:
    """Persistent cache shared across workspaces."""
    return Path.home() / ".cache" / "gnn_benchmark" / "lamah_ce"


def ensure_lamah_data_root(
    resolution: str,
    cache_dir: Path | None = None,
) -> Path:
    """Return a local LamaH-CE data_root, downloading + extracting if needed.

    Args:
        resolution: "daily" or "hourly". Only used to locate the matching
            time-series folder inside the already-extracted archive.
        cache_dir: Override the default cache location.

    Returns:
        Path to the extracted LamaH-CE root (the folder containing
        ``D_gauges/``, ``B_basins_intermediate_all/``, …).
    """
    if resolution not in {"daily", "hourly"}:
        raise ValueError(
            f"resolution must be 'daily' or 'hourly', got {resolution!r}"
        )
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    tarball_path = cache_root / _LAMAH_TARBALL_NAME
    extract_dir = cache_root / "extracted"
    marker = extract_dir / ".extracted_ok"

    if marker.exists():
        return _find_data_root(extract_dir)

    # Download (atomic: write to .part then rename)
    if not tarball_path.exists():
        if not GDOWN_AVAILABLE:
            raise ImportError(
                "gdown is required to fetch the LamaH-CE dataset. "
                "Install with: pip install gdown"
            )
        print(f"[LamaH-CE] Downloading archive from Google Drive -> {tarball_path}")
        print("[LamaH-CE] This is a large file; first-run fetch takes a while.")
        tmp = tarball_path.with_suffix(tarball_path.suffix + ".part")
        gdown.download(id=_LAMAH_GDRIVE_ID, output=str(tmp), quiet=False)
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
