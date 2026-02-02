"""I/O utility functions for saving and loading data."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_pickle(obj: Any, path: Path | str) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path | str) -> Any:
    """
    Load object from pickle file.

    Args:
        path: Input path

    Returns:
        Loaded object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_npz(
    path: Path | str,
    compressed: bool = True,
    **arrays: np.ndarray,
) -> None:
    """
    Save arrays to NPZ file.

    Args:
        path: Output path
        compressed: If True, use compression
        **arrays: Named arrays to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    """
    Load arrays from NPZ file.

    Args:
        path: Input path

    Returns:
        Dictionary of arrays
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    """
    Save object to JSON file.

    Args:
        obj: Object to save (must be JSON-serializable)
        path: Output path
        indent: Indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: Path | str) -> Any:
    """
    Load object from JSON file.

    Args:
        path: Input path

    Returns:
        Loaded object
    """
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: Path | str,
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """
    List files matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: If True, search recursively

    Returns:
        List of matching paths
    """
    directory = Path(directory)

    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))
