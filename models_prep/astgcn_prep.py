import pandas as pd
import numpy as np
import util
"""
Some info:
-ASTGCN cuts other features itself in load_graphdata_channel1, 
so even if we can pass multiple features in the numpy file, it ends up cutting 
all excluding the first feature=>Make sure the first feature is put first.
-If we put ourselves at the complete start of the pipeline, the code expects:
    -A npz file for the datapoints with shape (sequence_length, num_of_vertices, num_of_features)
    -An adjacency matrix is necessary and either expects an npy array as an adjacency matrix or a cost csv.
    =>It also needs an ID file of the node IDs in the second case.


TODO: Document how to actually run ASGCN with the necessary function calls
=>Basically, how do we pass these data files to the model afterwards.
"""


def series_csv_to_npz(series_csv: str, npz_path: str, value_col: str = "value1"):
    util.series_csv_to_npz(series_csv, npz_path, feature_cols=[value_col])

def write_sensor_ids(edges_csv: str, output_txt: str):
    util.write_sensor_ids(edges_csv, output_txt)
    