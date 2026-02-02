"""PEMS Speed dataset loaders (PEMS-BAY and METR-LA)."""

import pickle
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader

try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


METRLA_NODE_ORDER = [
    773869, 767541, 767542, 717447, 717446, 717445, 773062, 767620, 737529,
    717816, 765604, 767471, 716339, 773906, 765273, 716331, 771667, 716337,
    769953, 769402, 769403, 769819, 769405, 716941, 717578, 716960, 717804,
    767572, 767573, 773012, 773013, 764424, 769388, 716328, 717819, 769941,
    760987, 718204, 718045, 769418, 768066, 772140, 773927, 760024, 774012,
    774011, 767609, 769359, 760650, 716956, 769831, 761604, 717495, 716554,
    773953, 767470, 716955, 764949, 773954, 767366, 769444, 773939, 774067,
    769443, 767750, 767751, 767610, 773880, 764766, 717497, 717490, 717491,
    717492, 717493, 765176, 717498, 717499, 765171, 718064, 718066, 765164,
    769431, 769430, 717610, 767053, 767621, 772596, 772597, 767350, 767351,
    716571, 773023, 767585, 773024, 717483, 718379, 717481, 717480, 717486,
    764120, 772151, 718371, 717489, 717488, 717818, 718076, 718072, 767455,
    767454, 761599, 717099, 773916, 716968, 769467, 717576, 717573, 717572,
    717571, 717570, 764760, 718089, 769847, 717608, 767523, 716942, 718090,
    769867, 717472, 717473, 759591, 764781, 765099, 762329, 716953, 716951,
    767509, 765182, 769358, 772513, 716958, 718496, 769346, 773904, 718499,
    764853, 761003, 717502, 759602, 717504, 763995, 717508, 765265, 773996,
    773995, 717469, 717468, 764106, 717465, 764794, 717466, 717461, 717460,
    717463, 717462, 769345, 716943, 772669, 717582, 717583, 717580, 716949,
    717587, 772178, 717585, 716939, 768469, 764101, 767554, 773975, 773974,
    717510, 717513, 717825, 767495, 767494, 717821, 717823, 717458, 717459,
    769926, 764858, 717450, 717452, 717453, 759772, 717456, 771673, 772167,
    769372, 774204, 769806, 717590, 717592, 717595, 772168, 718141, 769373,
]

PEMS_BAY_NODE_ORDER = [
    400001, 400017, 400030, 400040, 400045, 400052, 400057, 400059, 400065,
    400069, 400073, 400084, 400085, 400088, 400096, 400097, 400100, 400104,
    400109, 400122, 400147, 400148, 400149, 400158, 400160, 400168, 400172,
    400174, 400178, 400185, 400201, 400206, 400209, 400213, 400221, 400222,
    400227, 400236, 400238, 400240, 400246, 400253, 400257, 400258, 400268,
    400274, 400278, 400280, 400292, 400296, 400298, 400330, 400336, 400343,
    400353, 400372, 400394, 400400, 400414, 400418, 400429, 400435, 400436,
    400440, 400449, 400457, 400461, 400464, 400479, 400485, 400499, 400507,
    400508, 400514, 400519, 400528, 400545, 400560, 400563, 400567, 400581,
    400582, 400586, 400637, 400643, 400648, 400649, 400654, 400664, 400665,
    400668, 400673, 400677, 400687, 400688, 400690, 400700, 400709, 400713,
    400714, 400715, 400717, 400723, 400743, 400750, 400760, 400772, 400790,
    400792, 400794, 400799, 400804, 400822, 400823, 400828, 400832, 400837,
    400842, 400863, 400869, 400873, 400895, 400904, 400907, 400911, 400916,
    400922, 400934, 400951, 400952, 400953, 400964, 400965, 400970, 400971,
    400973, 400995, 400996, 401014, 401129, 401154, 401163, 401167, 401210,
    401224, 401327, 401351, 401388, 401391, 401400, 401403, 401440, 401457,
    401464, 401489, 401495, 401507, 401534, 401541, 401555, 401560, 401567,
    401597, 401606, 401611, 401655, 401808, 401809, 401810, 401811, 401816,
    401817, 401845, 401846, 401890, 401891, 401906, 401908, 401926, 401936,
    401937, 401942, 401943, 401948, 401957, 401958, 401994, 401996, 401997,
    401998, 402056, 402057, 402058, 402059, 402060, 402061, 402067, 402117,
    402118, 402119, 402120, 402121, 402281, 402282, 402283, 402284, 402285,
    402286, 402287, 402288, 402289, 402359, 402360, 402361, 402362, 402363,
    402364, 402365, 402366, 402367, 402368, 402369, 402370, 402371, 402372,
    402373, 403225, 403265, 403329, 403401, 403402, 403404, 403406, 403409,
    403412, 403414, 403419, 404370, 404434, 404435, 404444, 404451, 404452,
    404453, 404461, 404462, 404521, 404522, 404553, 404554, 404585, 404586,
    404640, 404753, 404759, 405613, 405619, 405701, 407150, 407151, 407152,
    407153, 407155, 407157, 407161, 407165, 407172, 407173, 407174, 407176,
    407177, 407179, 407180, 407181, 407184, 407185, 407186, 407187, 407190,
    407191, 407194, 407200, 407202, 407204, 407206, 407207, 407321, 407323,
    407325, 407328, 407331, 407332, 407335, 407336, 407337, 407339, 407341,
    407342, 407344, 407348, 407352, 407359, 407360, 407361, 407364, 407367,
    407370, 407372, 407373, 407374, 407710, 407711, 408907, 408911, 409524,
    409525, 409526, 409528, 409529, 413026, 413845, 413877, 413878, 414284,
    414694,
]

# Google Drive IDs for data files
PEMS_BAY_GDRIVE_ID = "1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"
METRLA_GDRIVE_ID = "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"

# URLs for adjacency matrices
PEMS_BAY_ADJ_URL = "https://raw.githubusercontent.com/chnsh/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl"
METRLA_ADJ_URL = "https://raw.githubusercontent.com/chnsh/DCRNN/master/data/sensor_graph/adj_mx.pkl"


@dataclass
class PEMSBayLoader(DatasetLoader):
    """
    Loader for PEMS-BAY traffic speed dataset.

    Contains 5-minute traffic speed data from 325 sensors in the Bay Area.
    Requires gdown package for downloading from Google Drive.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="pems_bay",
            url=f"https://drive.google.com/uc?id={PEMS_BAY_GDRIVE_ID}",
            frequency="5T",
            node_order=[str(n) for n in PEMS_BAY_NODE_ORDER],
            feature_columns=["value"],
            units={"value": "mph"},
            description="PEMS-BAY traffic speed data (325 sensors)",
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download and convert PEMS-BAY data."""
        if not GDOWN_AVAILABLE:
            raise ImportError(
                "gdown is required for PEMS-BAY dataset. "
                "Install with: pip install gdown"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            h5_path = tmpdir / "pems-bay.h5"
            adj_path = tmpdir / "adj_mx_bay.pkl"

            # Download data file from Google Drive
            print("Downloading PEMS-BAY data from Google Drive...")
            gdown.download(id=PEMS_BAY_GDRIVE_ID, output=str(h5_path), quiet=False)

            # Download adjacency matrix
            print("Downloading adjacency matrix...")
            urllib.request.urlretrieve(PEMS_BAY_ADJ_URL, str(adj_path))

            # Convert to IR format
            series_df = self._convert_series(h5_path, PEMS_BAY_NODE_ORDER)
            edges_df = self._convert_adjacency(adj_path)

            return series_df, edges_df

    def _convert_series(self, h5_path: Path, node_order: list[int]) -> pd.DataFrame:
        """Convert HDF5 data to series DataFrame."""
        with pd.HDFStore(h5_path, mode="r") as store:
            key = store.keys()[0]
            wide = store.get(key)

        wide.index.name = "ts"

        # Convert to long format
        long = (
            wide.stack()
            .rename_axis(["ts", "node_id"])
            .reset_index(name="value")
        )

        # Replace zeros with NaN
        long["value"] = long["value"].where(long["value"] != 0)

        # Ensure types
        long["ts"] = pd.to_datetime(long["ts"], errors="coerce")
        long["node_id"] = long["node_id"].astype("int64")

        # Densify to full grid
        ts_min, ts_max = long["ts"].min(), long["ts"].max()
        full_ts = pd.date_range(ts_min, ts_max, freq="5min")
        node_idx = pd.Index(node_order, dtype="int64")

        grid = pd.MultiIndex.from_product(
            [full_ts, node_idx], names=["ts", "node_id"]
        ).to_frame(index=False)

        long = (
            grid.merge(long, on=["ts", "node_id"], how="left")
            .sort_values(["ts", "node_id"])
            .reset_index(drop=True)
        )

        # Convert node_id to string
        long["node_id"] = long["node_id"].astype(str)

        return long

    def _convert_adjacency(self, adj_path: Path) -> pd.DataFrame:
        """Convert pickle adjacency matrix to edges DataFrame."""
        with open(adj_path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")

        ids, id_to_ind, adj = obj[0], obj[1], np.asarray(obj[2])

        # Extract non-zero, non-diagonal edges
        i, j = np.nonzero(adj)
        mask = i != j

        edges = pd.DataFrame({
            "src": [str(ids[a]) for a in i[mask]],
            "dst": [str(ids[b]) for b in j[mask]],
            "cost": adj[i[mask], j[mask]].astype(float),
        })

        return edges


@dataclass
class MetroLALoader(DatasetLoader):
    """
    Loader for METR-LA traffic speed dataset.

    Contains 5-minute traffic speed data from 207 sensors in Los Angeles.
    Requires gdown package for downloading from Google Drive.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="metr_la",
            url=f"https://drive.google.com/uc?id={METRLA_GDRIVE_ID}",
            frequency="5T",
            node_order=[str(n) for n in METRLA_NODE_ORDER],
            feature_columns=["value"],
            units={"value": "mph"},
            description="METR-LA traffic speed data (207 sensors)",
        )

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download and convert METR-LA data."""
        if not GDOWN_AVAILABLE:
            raise ImportError(
                "gdown is required for METR-LA dataset. "
                "Install with: pip install gdown"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            h5_path = tmpdir / "metr-la.h5"
            adj_path = tmpdir / "adj_mx.pkl"

            # Download data file from Google Drive
            print("Downloading METR-LA data from Google Drive...")
            gdown.download(id=METRLA_GDRIVE_ID, output=str(h5_path), quiet=False)

            # Download adjacency matrix
            print("Downloading adjacency matrix...")
            urllib.request.urlretrieve(METRLA_ADJ_URL, str(adj_path))

            # Convert to IR format
            series_df = self._convert_series(h5_path, METRLA_NODE_ORDER)
            edges_df = self._convert_adjacency(adj_path)

            return series_df, edges_df

    def _convert_series(self, h5_path: Path, node_order: list[int]) -> pd.DataFrame:
        """Convert HDF5 data to series DataFrame."""
        with pd.HDFStore(h5_path, mode="r") as store:
            key = store.keys()[0]
            wide = store.get(key)

        wide.index.name = "ts"

        # Convert to long format
        long = (
            wide.stack()
            .rename_axis(["ts", "node_id"])
            .reset_index(name="value")
        )

        # Replace zeros with NaN
        long["value"] = long["value"].where(long["value"] != 0)

        # Ensure types
        long["ts"] = pd.to_datetime(long["ts"], errors="coerce")
        long["node_id"] = long["node_id"].astype("int64")

        # Densify to full grid
        ts_min, ts_max = long["ts"].min(), long["ts"].max()
        full_ts = pd.date_range(ts_min, ts_max, freq="5min")
        node_idx = pd.Index(node_order, dtype="int64")

        grid = pd.MultiIndex.from_product(
            [full_ts, node_idx], names=["ts", "node_id"]
        ).to_frame(index=False)

        long = (
            grid.merge(long, on=["ts", "node_id"], how="left")
            .sort_values(["ts", "node_id"])
            .reset_index(drop=True)
        )

        # Convert node_id to string
        long["node_id"] = long["node_id"].astype(str)

        return long

    def _convert_adjacency(self, adj_path: Path) -> pd.DataFrame:
        """Convert pickle adjacency matrix to edges DataFrame."""
        with open(adj_path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")

        ids, id_to_ind, adj = obj[0], obj[1], np.asarray(obj[2])

        # Extract non-zero, non-diagonal edges
        i, j = np.nonzero(adj)
        mask = i != j

        edges = pd.DataFrame({
            "src": [str(ids[a]) for a in i[mask]],
            "dst": [str(ids[b]) for b in j[mask]],
            "cost": adj[i[mask], j[mask]].astype(float),
        })

        return edges
