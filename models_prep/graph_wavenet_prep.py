import util
import pandas as pd
import numpy as np
"""
Some info:
-GraphWaveNet appears to only take in one value per node at the generate_data step, so if we want to intake more we need to modify the code,
or to directly supply train
-If we put ourselves at the complete start of the pipeline, the code expects:
    -For the time-series data, an h5 file
    -For the adjacency matrix, which is optional, a pickle file
Both functions to generate these from our intermediate representation are below
Example usage:
series_csv_to_h5("series.csv", "data/mydata.h5")
edges_csv_to_adj("edges.csv", "data/adj_mx.pkl")

TODO: Document how to actually run graph_wavenet with the necessary function calls
=>Basically, how do we pass these data files to the model afterwards.
"""

def series_csv_to_h5(series_csv: str, h5_path: str, value_col=0):
    """Convert series.csv into the HDF5 format expected by generate_training_data.py."""
    util.series_csv_to_h5(series_csv, h5_path, value_col)

def edges_csv_to_adj(path: str, pkl_path: str, model_name):
    util.edges_csv_to_adj(path, pkl_path, model_name)