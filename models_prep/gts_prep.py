import util

"""
Some info:
-GTS appears to only take in one value per node, so if we want to intake more we need to modify the code.
-GTS accepts an adjacency matrix. =>Their CSV format for the adjacency matrix is the same as ours, 
however they also expect a list of names.
There is also the possibility (could maybe get better results for non-traffic datasets) of directly supplying
our own adjacency matrix in a pickle file.


TODO: Document how to actually run GTS with the necessary function calls
=>Basically, how do we pass these data files to the model afterwards.
=>Will also need to take care of the YAML file.
"""

def series_csv_to_h5(series_csv: str, h5_path: str):
    """Convert series.csv into the HDF5 format expected by generate_training_data.py."""
    util.series_csv_to_h5(series_csv, h5_path)


def write_sensor_ids(edges_csv: str, output_txt: str):
    util.write_sensor_ids(edges_csv, output_txt)
    