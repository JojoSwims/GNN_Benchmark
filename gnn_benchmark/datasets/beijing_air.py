"""Beijing Air Quality dataset loader."""

import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

from gnn_benchmark.core.types import DatasetInfo
from gnn_benchmark.datasets.base import DatasetLoader

# Node orders for different subdivisions
BEIJING_NODE_ORDER = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
    1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030,
    1031, 1032, 1033, 1034, 1035, 1036,
]

CLUSTER1_NODE_ORDER = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
    1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030,
    1031, 1032, 1033, 1034, 1035, 1036, 6001, 6002, 6003, 6004,
    6005, 6006, 6007, 6008, 6010, 6011, 6012, 6013, 6014, 6015,
    6016, 6017, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026,
    6027, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008,
    11009, 11010, 11011, 11012, 11013, 11014, 11015, 11016, 11017,
    11018, 11019, 11020, 11021, 11022, 11023, 11024, 11025, 12001,
    13001, 13003, 13004, 13005, 13006, 13007, 13008, 13009, 13010,
    13011, 13012, 13013, 13014, 14001, 14002, 14003, 14004, 14005,
    14006, 14007, 14008, 17001, 17002, 17003, 17004, 17005, 17007,
    17008, 17009, 17010, 17011, 17012, 18001, 19001, 19002, 19003,
    19004, 19005, 19006, 19007, 19008, 19009, 19010, 20001, 20002,
    20003, 20004, 20005, 20006, 20007, 20008, 20009, 21001, 21002,
    21003, 21004, 21005, 21006, 21007, 22001, 22002, 22003, 22004,
    22005, 22006, 22007, 22008, 22009, 22010, 23001, 23002, 23003,
    23004, 23005, 6028, 13002, 17006, 128001, 128002, 128003, 128004,
    128005, 128006, 128007, 149001, 149004, 149006, 149007, 149009,
    149010, 149014, 149015, 151003, 151004, 151007, 151009, 151011,
    152001, 152004, 159001, 159002, 159003, 160001, 160005, 160006,
    152002, 159004, 160003, 160004, 160007, 160008, 160009, 159005,
    151002, 151005, 151006, 151008, 151010, 151012, 151013, 152003,
    152005, 152006, 159006, 13015, 13016, 13017, 13018, 14009, 17013,
    17014, 17015, 17016, 17017, 17018, 17019, 17020, 17021, 17022,
    17023, 17024, 17025, 17026, 17027, 17028, 19011, 19012, 19013,
    19014, 19015, 19016, 19017, 19018, 19019, 19020, 20010, 20011,
    20012, 20013, 20014, 21008, 21009, 21010, 21011, 21012, 21013,
    21014, 21015, 21016, 22011, 22012, 23006, 23007, 23008, 23009,
    23010, 23011, 23012, 23013, 149002, 149003, 149005, 149008,
    149011, 149012, 149013, 6040,
]

CLUSTER2_NODE_ORDER = [
    4003, 4007, 4008, 4009, 4011, 4014, 4017, 4018, 4019, 4020,
    9017, 9018, 9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026,
    9027, 9028, 9029, 9030, 9031, 9032, 9033, 9034, 9035, 9036,
    9037, 9038, 9039, 9040, 9041, 9042, 9043, 9044, 9045, 9046,
    10001, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010,
    10011, 10012, 10013, 10014, 10015, 10016, 24001, 24002, 24003,
    25001, 25002, 25003, 25004, 26001, 26002, 26003, 26004, 26005,
    26006, 26007, 26008, 27001, 27002, 27003, 28001, 28002, 28003,
    28004, 28005, 28006, 28007, 29001, 29003, 29004, 29005, 29006,
    29007, 30002, 30003, 30004, 31001, 31002, 31003, 31004, 32001,
    32002, 32003, 33001, 33002, 33003, 34001, 34002, 34003, 34004,
    34005, 34006, 35001, 35002, 35003, 36001, 36002, 36003, 36004,
    36005, 37001, 37002, 37003, 38001, 38002, 38003, 40001, 40002,
    40003, 41001, 41002, 41003, 41004, 42001, 42002, 42003, 42004,
    9058, 9047, 4002, 9016, 9059, 9060, 9061, 9062, 371001, 371002,
    371003, 371004, 372001, 372002, 311001, 311002, 311003, 311004,
    311005, 9064, 9065, 9066, 9067, 25006, 27004, 27005, 27006,
    30005, 30006, 30007, 33004, 9063, 29008,
]

URL = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Data-1.zip"


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


@dataclass
class BeijingAirLoader(DatasetLoader):
    """
    Loader for Beijing Air Quality dataset.

    Downloads PM2.5 concentration data for measuring stations in the
    Beijing-Tianjin-Shenzhen-Guangzhou area.

    Args:
        subdivision: Which subset of stations to use:
            - "beijing": Only Beijing city stations (36 nodes)
            - "cluster1": Beijing-Tianjin region (284 nodes)
            - "cluster2": Shenzhen-Guangzhou region (163 nodes)
            - "all": All stations (437 nodes)
    """

    subdivision: str = "beijing"

    def __post_init__(self):
        valid_subdivisions = {"beijing", "cluster1", "cluster2", "all"}
        if self.subdivision not in valid_subdivisions:
            raise ValueError(
                f"Invalid subdivision '{self.subdivision}'. "
                f"Must be one of: {valid_subdivisions}"
            )

    @property
    def info(self) -> DatasetInfo:
        node_order = self._get_node_order()
        name = f"beijing_air_{self.subdivision}"
        if self.subdivision == "beijing":
            name = "beijing_air"

        return DatasetInfo(
            name=name,
            url=URL,
            frequency="1H",
            node_order=[str(n) for n in node_order],
            feature_columns=["PM25_Concentration"],
            units={"PM25_Concentration": "ug/m3"},
            description=f"Beijing Air Quality - {self.subdivision} subdivision",
        )

    def _get_node_order(self) -> list[int]:
        if self.subdivision == "beijing":
            return BEIJING_NODE_ORDER
        elif self.subdivision == "cluster1":
            return CLUSTER1_NODE_ORDER
        elif self.subdivision == "cluster2":
            return CLUSTER2_NODE_ORDER
        else:  # all
            return CLUSTER1_NODE_ORDER + CLUSTER2_NODE_ORDER

    def download_and_convert(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Download and convert Beijing Air Quality data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "data.zip"
            extract_dir = tmpdir / "extracted"

            # Download
            print(f"Downloading Beijing Air Quality data...")
            urllib.request.urlretrieve(URL, str(zip_path))

            # Extract
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Find the Data folder (may be nested)
            data_dir = self._find_data_dir(extract_dir)

            # Get node list based on subdivision
            node_ids = self._get_node_ids(data_dir)

            # Convert to IR format
            series_df = self._convert_series(data_dir, node_ids)
            edges_df = self._compute_edges(data_dir, node_ids)

            return series_df, edges_df

    def _find_data_dir(self, extract_dir: Path) -> Path:
        """Find the Data directory in extracted files."""
        # Look for Data folder at various levels
        for pattern in ["Data", "*/Data", "*/*/Data"]:
            matches = list(extract_dir.glob(pattern))
            if matches:
                return matches[0]

        # If no Data folder, look for airquality.csv directly
        for pattern in ["airquality.csv", "*/airquality.csv", "*/*/airquality.csv"]:
            matches = list(extract_dir.glob(pattern))
            if matches:
                return matches[0].parent

        raise FileNotFoundError("Could not find data directory in extracted files")

    def _get_node_ids(self, data_dir: Path) -> list[int]:
        """Get node IDs for the selected subdivision."""
        stations = pd.read_csv(data_dir / "station.csv")

        if self.subdivision == "beijing":
            district_ids = list(range(101, 117))  # Beijing districts
            ids = stations.loc[
                stations["district_id"].isin(district_ids), "station_id"
            ].tolist()

        elif self.subdivision in ("cluster1", "cluster2"):
            cluster_number = int(self.subdivision[-1])
            cities = pd.read_csv(data_dir / "city.csv")
            city_ids = cities.loc[
                cities["cluster_id"] == cluster_number, "city_id"
            ].tolist()

            districts = pd.read_csv(data_dir / "district.csv")
            district_ids = districts.loc[
                districts["city_id"].isin(city_ids), "district_id"
            ].tolist()

            ids = stations.loc[
                stations["district_id"].isin(district_ids), "station_id"
            ].tolist()

        else:  # all
            ids = stations["station_id"].tolist()

        return ids

    def _convert_series(self, data_dir: Path, node_ids: list[int]) -> pd.DataFrame:
        """Convert air quality data to series DataFrame."""
        df = pd.read_csv(data_dir / "airquality.csv", parse_dates=[1])
        node_col, ts_col = df.columns[0], df.columns[1]

        # Filter to selected nodes
        df = df[df[node_col].isin(node_ids)]

        # Remove invalid dates
        df = df[df[ts_col] > pd.Timestamp("2014-01-01")]

        # Keep only PM2.5
        feat_cols = ["PM25_Concentration"]
        df = df[[ts_col, node_col, *feat_cols]].copy()
        df.rename(columns={ts_col: "ts", node_col: "node_id"}, inplace=True)

        # Treat zeros as missing
        df.loc[:, feat_cols] = df.loc[:, feat_cols].where(df.loc[:, feat_cols] != 0)

        # Densify to full grid
        ts_unique = pd.Index(sorted(df["ts"].dropna().unique()))
        full_ts = pd.date_range(ts_unique[0], ts_unique[-1], freq="1H")

        enforced_order = self._get_node_order()
        # Filter enforced_order to only include nodes that exist
        enforced_order = [n for n in enforced_order if n in node_ids]

        grid = pd.MultiIndex.from_product(
            [full_ts, pd.Index(enforced_order, dtype="int64")],
            names=["ts", "node_id"],
        ).to_frame(index=False)

        merged = grid.merge(df, on=["ts", "node_id"], how="left")
        merged = merged.sort_values(["ts", "node_id"]).reset_index(drop=True)

        # Convert node_id to string
        merged["node_id"] = merged["node_id"].astype(str)

        return merged

    def _compute_edges(self, data_dir: Path, node_ids: list[int]) -> pd.DataFrame:
        """Compute edge weights based on distances between stations."""
        stations = pd.read_csv(
            data_dir / "station.csv",
            dtype={"station_id": "int64", "latitude": "float64", "longitude": "float64"},
        )
        stations = stations[stations["station_id"].isin(node_ids)]
        stations = stations[["station_id", "latitude", "longitude"]].reset_index(drop=True)

        rows = []
        for i, row_i in stations.iterrows():
            for j, row_j in stations.iterrows():
                if row_i["station_id"] == row_j["station_id"]:
                    continue
                d = _haversine_distance(
                    row_i["latitude"],
                    row_i["longitude"],
                    row_j["latitude"],
                    row_j["longitude"],
                )
                rows.append((str(row_i["station_id"]), str(row_j["station_id"]), d))

        edges = pd.DataFrame(rows, columns=["src", "dst", "cost"])
        return edges
