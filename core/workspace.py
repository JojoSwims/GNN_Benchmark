"""DataWorkspace class for managing data files."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gnn_benchmark.datasets.base import DatasetLoader

from gnn_benchmark.core.intermediate import IntermediateRepresentation


class DataWorkspace:
    """
    Central manager for all data operations.

    Handles downloading, caching, and organizing datasets.
    Users point to a single workspace directory for all operations.

    Directory structure:
        workspace/
        ├── clean/          # Clean IR after conversion (never modified)
        ├── working/        # Current transformed state (mutable)
        └── exports/        # Model-specific outputs
    """

    def __init__(self, path: Path | str):
        """
        Initialize a workspace.

        Args:
            path: Path to workspace directory. Will be created if it doesn't exist.
        """
        self.path = Path(path)
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Create workspace directory structure if it doesn't exist."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(exist_ok=True)
        self.working_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)

    # --- Directory accessors ---

    @property
    def clean_dir(self) -> Path:
        """Directory for clean (original) intermediate representations."""
        return self.path / "clean"

    @property
    def working_dir(self) -> Path:
        """Directory for working (transformed) intermediate representations."""
        return self.path / "working"

    @property
    def exports_dir(self) -> Path:
        """Directory for model-specific exports."""
        return self.path / "exports"

    # --- Dataset management ---

    def is_downloaded(self, dataset_name: str) -> bool:
        """
        Check if a dataset has been downloaded and converted.

        A dataset is considered downloaded if it has a clean IR.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if clean IR exists for this dataset
        """
        clean_path = self.clean_dir / dataset_name
        return (clean_path / "series.csv").exists() and (clean_path / "metadata.json").exists()

    def has_working(self, dataset_name: str) -> bool:
        """
        Check if a dataset has a working (transformed) version.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if working IR exists for this dataset
        """
        working_path = self.working_dir / dataset_name
        return (working_path / "series.csv").exists()

    def has_exports(self, dataset_name: str) -> bool:
        """
        Check if a dataset has any exports.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if exports directory exists and is non-empty
        """
        export_path = self.exports_dir / dataset_name
        return export_path.exists() and any(export_path.iterdir())

    def list_datasets(self) -> list[str]:
        """
        List all datasets with clean IR in this workspace.

        Returns:
            List of dataset names
        """
        datasets = []
        for d in self.clean_dir.iterdir():
            if d.is_dir() and (d / "series.csv").exists():
                datasets.append(d.name)
        return sorted(datasets)

    def list_exports(self, dataset_name: str) -> list[str]:
        """
        List all exports for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of exporter names (e.g., ["staeformer", "graph_wavenet"])
        """
        export_path = self.exports_dir / dataset_name
        if not export_path.exists():
            return []
        return sorted([d.name for d in export_path.iterdir() if d.is_dir()])

    # --- Loading ---

    def load(
        self, dataset_name: str, from_clean: bool = False
    ) -> IntermediateRepresentation:
        """
        Load a dataset's intermediate representation.

        Args:
            dataset_name: Name of dataset to load
            from_clean: If True, load from clean/ (ignoring working state).
                       If False, load from working/ if exists, else from clean/.

        Returns:
            IntermediateRepresentation linked to this workspace

        Raises:
            FileNotFoundError: If dataset doesn't exist in workspace
        """
        if from_clean:
            path = self.clean_dir / dataset_name
        else:
            # Prefer working, fall back to clean
            working_path = self.working_dir / dataset_name
            if (working_path / "series.csv").exists():
                path = working_path
            else:
                path = self.clean_dir / dataset_name

        if not (path / "series.csv").exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found in workspace. "
                f"Use a DatasetLoader to prepare it first."
            )

        return IntermediateRepresentation.load(
            path, workspace=self, dataset_name=dataset_name
        )

    # --- Preparation ---

    def prepare(
        self,
        loader: DatasetLoader,
        force_download: bool = False,
    ) -> IntermediateRepresentation:
        """
        Prepare a dataset using the given loader.

        1. Download and convert to clean IR (if not cached or force_download)
        2. Copy clean to working
        3. Return IR linked to workspace

        Args:
            loader: DatasetLoader instance to use
            force_download: If True, re-download even if cached

        Returns:
            IntermediateRepresentation linked to this workspace
        """
        dataset_name = loader.info.name

        # Check if already downloaded
        if not self.is_downloaded(dataset_name) or force_download:
            # Download and convert
            series_df, edges_df = loader.download_and_convert()

            # Create metadata from loader info
            from gnn_benchmark.core.types import IRMetadata

            metadata = IRMetadata(
                name=loader.info.name,
                frequency=loader.info.frequency,
                node_order=loader.info.node_order,
                feature_columns=loader.info.feature_columns,
                units=loader.info.units,
                source_url=loader.info.url,
                transform_history=[],
            )

            # Create IR and save to clean
            ir = IntermediateRepresentation(
                series=series_df,
                metadata=metadata,
                edges=edges_df,
                workspace=self,
                dataset_name=dataset_name,
            )
            ir.save(self.clean_dir / dataset_name)

        # Copy clean to working (always refresh working from clean)
        clean_path = self.clean_dir / dataset_name
        working_path = self.working_dir / dataset_name

        if working_path.exists():
            shutil.rmtree(working_path)
        shutil.copytree(clean_path, working_path)

        # Load and return from working
        return self.load(dataset_name, from_clean=False)

    def save_working(self, ir: IntermediateRepresentation) -> Path:
        """
        Save the current state of an IR to the working directory.

        Args:
            ir: IntermediateRepresentation to save

        Returns:
            Path where data was saved

        Raises:
            ValueError: If IR is not linked to this workspace
        """
        if ir.workspace is not self or ir.dataset_name is None:
            raise ValueError("IR is not linked to this workspace")

        return ir.save(self.working_dir / ir.dataset_name)

    def clear_working(self, dataset_name: str) -> None:
        """
        Remove working state for a dataset, resetting to clean.

        Args:
            dataset_name: Name of the dataset
        """
        working_path = self.working_dir / dataset_name
        if working_path.exists():
            shutil.rmtree(working_path)

    def clear_exports(self, dataset_name: str, exporter_name: str | None = None) -> None:
        """
        Remove exports for a dataset.

        Args:
            dataset_name: Name of the dataset
            exporter_name: If provided, only clear exports for this exporter.
                          If None, clear all exports for the dataset.
        """
        if exporter_name:
            export_path = self.exports_dir / dataset_name / exporter_name
        else:
            export_path = self.exports_dir / dataset_name

        if export_path.exists():
            shutil.rmtree(export_path)

    def __repr__(self) -> str:
        datasets = self.list_datasets()
        return f"DataWorkspace(path={self.path!r}, datasets={datasets})"
