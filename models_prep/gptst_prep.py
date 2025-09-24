import util
"""
Info:
    -Model accepts multiple features!!
    -Features are stored in an npz file.
    -Appears to not require an adjacency matrix (to check!)
#TODO:Check final how to actually call model
"""

def series_csv_to_npz(series_csv: str, npz_path: str):
    util.series_csv_to_npz(series_csv, npz_path)