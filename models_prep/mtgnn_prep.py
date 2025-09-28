import util
import pandas as pd
"""
Some info:
-MTGNN appears to only take in one value per node, so if we want to intake more we need to modify the code.
-MTGNN differentiates between Single-Step training (predict the next value) and multi-step training (predicts, say 6 values ahead) and the code is somewhat different for both.
In practice, both have a toggle, buildA_true which decides if we build an adjacency matrix or not. However there is no code to provide an Adjacency matrix for single-step (would need to modify the code)
and in multi-step, an adjacency matrix has to be provided (which is potentially discarded if buildA_true is set to True).
=>So edges_csv_to_adj is only necessary to be called for multi_step training.

As for converting the series.csv file, in multi-step we go through the generate_training_data.py function.
and in single-step we do not, so the routine is different.

TODO: Document how to actually run MTGNN with the necessary function calls
=>Basically, how do we pass these data files to the model afterwards.
"""


#Functions for multi-step-training:

def series_csv_to_h5(series_csv: str, h5_path: str):
    """Convert series.csv into the HDF5 format expected by generate_training_data.py."""
    util.series_csv_to_h5(series_csv, h5_path)

def edges_csv_to_adj(edges_csv: str, pkl_path: str):
    util.edges_csv_to_adj(edges_csv, pkl_path)

#--------------------------------------------------------------------------------------

#Function for single-step training.
def series_csv_to_txt(series_csv: str, txt_path: str, value_col: str = util.FIRST_COLUMN_NAME):
    """
    Convert `series.csv` to the plain matrix format used by `train_single_step.py`.
    Input CSV columns
        ts, node_id, value1[, value2, ...]
    Output TXT
        rows = chronologically‑sorted timestamps
        cols = nodes ordered by `node_id` (ascending)
    """
    df = pd.read_csv(series_csv, parse_dates=["ts"])
    df.rename(columns={df.columns[0]: "ts", df.columns[1]: "node_id"}, inplace=True)

    table = (
        df.pivot(index="ts", columns="node_id", values=value_col)
          .sort_index()
    )
    table.to_csv(txt_path, header=False, index=False)