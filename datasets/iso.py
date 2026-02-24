
import urllib.request
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mis.nyiso.com/public/csv/palIntegrated"
START_YEAR = 2012
END_YEAR = 2017
OUT_MERGED = Path("palIntegrated_2012_2017_merged.csv")

#Keep this even if we refactor, it is currently not used but will be used 
PTID_TO_NAME_DICT={
    61757: "CAPITL",
    61754: "CENTRL",
    61760: "DUNWOD",
    61753: "GENESE",
    61758: "HUD VL",
    61762: "LONGIL",
    61756: "MHK VL",
    61759: "MILLWD",
    61761: "N.Y.C.",
    61755: "NORTH",
    61752: "WEST"
}

"Info: 60 minute interval"

if __name__ == "__main__":
    frames = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # Download all monthly ZIPs, then read all daily CSVs inside
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                fname = f"{year}{month:02d}01palIntegrated_csv.zip"
                url = f"{BASE_URL}/{fname}"
                zpath = tmpdir / fname

                urllib.request.urlretrieve(url, zpath)

                with zipfile.ZipFile(zpath) as zf:
                    for member in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
                        with zf.open(member) as f:
                            frames.append(pd.read_csv(f))

    # Concatenate all rows
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamps
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    df["Time Zone"] = df["Time Zone"].astype(str).str.strip()

    # Keep only requested local-date range
    start_local = pd.Timestamp(f"{START_YEAR}-01-01 00:00:00")
    end_local = pd.Timestamp(f"{END_YEAR}-12-31 23:00:00")
    df = df[(df["Time Stamp"] >= start_local) & (df["Time Stamp"] <= end_local)].copy()

    # Build canonical UTC timestamp from local timestamp + explicit EST/EDT tag
    tz_to_offset = {"EST": "-0500", "EDT": "-0400"}
    ts_str = df["Time Stamp"].dt.strftime("%m/%d/%Y %H:%M:%S")
    df["ts_utc"] = pd.to_datetime(
        ts_str + " " + df["Time Zone"].map(tz_to_offset),
        format="%m/%d/%Y %H:%M:%S %z",
        utc=True,
    )

    # Clean load values, keep missing values as NA, round valid values to whole numbers
    load = pd.to_numeric(
        df["Integrated Load"].astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    ).replace([np.inf, -np.inf], pd.NA)
    df["Integrated Load"] = load.round().astype("Int64")

    # Final output (drop Time Zone and Name)
    out = df[["ts_utc", "Time Stamp", "PTID", "Integrated Load"]].copy()
    out = out.sort_values(["ts_utc", "PTID"]).reset_index(drop=True)
    out.to_csv(OUT_MERGED, index=False)