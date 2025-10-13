import pandas as pd
from pathlib import Path
import numpy as np
import gdown
import pickle
import shutil
import urllib.request

"""
Important note on the adjacency matrix. For some reason, the adjacency matrices provided are not 
truly symetric. 
"""
METRLA_NODE_ORDER=[773869,767541,767542,717447,717446,717445,773062,767620,737529,717816,765604,767471,716339,773906,765273,716331,771667,716337,769953,769402,769403,769819,769405,716941,717578,716960,717804,767572,767573,773012,773013,764424,769388,716328,717819,769941,760987,718204,718045,769418,768066,772140,773927,760024,774012,774011,767609,769359,760650,716956,769831,761604,717495,716554,773953,767470,716955,764949,773954,767366,769444,773939,774067,769443,767750,767751,767610,773880,764766,717497,717490,717491,717492,717493,765176,717498,717499,765171,718064,718066,765164,769431,769430,717610,767053,767621,772596,772597,767350,767351,716571,773023,767585,773024,717483,718379,717481,717480,717486,764120,772151,718371,717489,717488,717818,718076,718072,767455,767454,761599,717099,773916,716968,769467,717576,717573,717572,717571,717570,764760,718089,769847,717608,767523,716942,718090,769867,717472,717473,759591,764781,765099,762329,716953,716951,767509,765182,769358,772513,716958,718496,769346,773904,718499,764853,761003,717502,759602,717504,763995,717508,765265,773996,773995,717469,717468,764106,717465,764794,717466,717461,717460,717463,717462,769345,716943,772669,717582,717583,717580,716949,717587,772178,717585,716939,768469,764101,767554,773975,773974,717510,717513,717825,767495,767494,717821,717823,717458,717459,769926,764858,717450,717452,717453,759772,717456,771673,772167,769372,774204,769806,717590,717592,717595,772168,718141,769373]
PEMS_BAY_NODE_ORDER=[400001, 400017, 400030, 400040, 400045, 400052, 400057, 400059, 400065, 400069, 400073, 400084, 400085, 400088, 400096, 400097, 400100, 400104, 400109, 400122, 400147, 400148, 400149, 400158, 400160, 400168, 400172, 400174, 400178, 400185, 400201, 400206, 400209, 400213, 400221, 400222, 400227, 400236, 400238, 400240, 400246, 400253, 400257, 400258, 400268, 400274, 400278, 400280, 400292, 400296, 400298, 400330, 400336, 400343, 400353, 400372, 400394, 400400, 400414, 400418, 400429, 400435, 400436, 400440, 400449, 400457, 400461, 400464, 400479, 400485, 400499, 400507, 400508, 400514, 400519, 400528, 400545, 400560, 400563, 400567, 400581, 400582, 400586, 400637, 400643, 400648, 400649, 400654, 400664, 400665, 400668, 400673, 400677, 400687, 400688, 400690, 400700, 400709, 400713, 400714, 400715, 400717, 400723, 400743, 400750, 400760, 400772, 400790, 400792, 400794, 400799, 400804, 400822, 400823, 400828, 400832, 400837, 400842, 400863, 400869, 400873, 400895, 400904, 400907, 400911, 400916, 400922, 400934, 400951, 400952, 400953, 400964, 400965, 400970, 400971, 400973, 400995, 400996, 401014, 401129, 401154, 401163, 401167, 401210, 401224, 401327, 401351, 401388, 401391, 401400, 401403, 401440, 401457, 401464, 401489, 401495, 401507, 401534, 401541, 401555, 401560, 401567, 401597, 401606, 401611, 401655, 401808, 401809, 401810, 401811, 401816, 401817, 401845, 401846, 401890, 401891, 401906, 401908, 401926, 401936, 401937, 401942, 401943, 401948, 401957, 401958, 401994, 401996, 401997, 401998, 402056, 402057, 402058, 402059, 402060, 402061, 402067, 402117, 402118, 402119, 402120, 402121, 402281, 402282, 402283, 402284, 402285, 402286, 402287, 402288, 402289, 402359, 402360, 402361, 402362, 402363, 402364, 402365, 402366, 402367, 402368, 402369, 402370, 402371, 402372, 402373, 403225, 403265, 403329, 403401, 403402, 403404, 403406, 403409, 403412, 403414, 403419, 404370, 404434, 404435, 404444, 404451, 404452, 404453, 404461, 404462, 404521, 404522, 404553, 404554, 404585, 404586, 404640, 404753, 404759, 405613, 405619, 405701, 407150, 407151, 407152, 407153, 407155, 407157, 407161, 407165, 407172, 407173, 407174, 407176, 407177, 407179, 407180, 407181, 407184, 407185, 407186, 407187, 407190, 407191, 407194, 407200, 407202, 407204, 407206, 407207, 407321, 407323, 407325, 407328, 407331, 407332, 407335, 407336, 407337, 407339, 407341, 407342, 407344, 407348, 407352, 407359, 407360, 407361, 407364, 407367, 407370, 407372, 407373, 407374, 407710, 407711, 408907, 408911, 409524, 409525, 409526, 409528, 409529, 413026, 413845, 413877, 413878, 414284, 414694]
BAY_ID="1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"
METRLA_ID="1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
BAY_URL="https://raw.githubusercontent.com/chnsh/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl"
METRLA_URL="https://raw.githubusercontent.com/chnsh/DCRNN/master/data/sensor_graph/adj_mx.pkl"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/pems_speed")
default_set=[]

def _download(dataset_select, zip_path = ZIP_DEFAULT_PATH, out_dir = OUT_DEFAULT_PATH):
    out_dir.mkdir(parents=True, exist_ok=True)

    if "pemsbay" in dataset_select:
        gdown.download(id=BAY_ID, output=str(out_dir/"pems-bay.h5"), quiet=False)
        urllib.request.urlretrieve(BAY_URL, out_dir/"adj_mx_bay.pkl")

    if "metrla" in dataset_select:
        gdown.download(id=METRLA_ID, output=str(out_dir/"metr-la.h5"), quiet=False)
        urllib.request.urlretrieve(METRLA_URL, out_dir/"adj_mx.pkl")



def _cleanup(target_dir):
    """Delete files (not dirs) inside target_dir, move remaining items up one level, then remove target_dir."""
    p = Path(target_dir)
    if not p.exists():
        return
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    # Delete files and symlinks directly under target_dir
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()

    # Move remaining entries (typically directories) to the parent
    parent = p.parent
    for child in list(p.iterdir()):
        dest = parent / child.name
        if dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")
        shutil.move(str(child), str(dest))

    # 3) Remove now-empty target_dir
    p.rmdir()
    pass

def _individual_prepare_data(out_dir, dataset):
    if dataset=="metrla":
        fpath=out_dir/"metr-la.h5"
    elif dataset=="pemsbay":
        fpath=out_dir/"pems-bay.h5"
    else:
        raise NotImplementedError("An unaccepted dataset_select value was passed to prepare(), please pass a list containing one or both of 'metrla' and 'pemsbay")
        
    with pd.HDFStore(fpath, mode="r") as store:
        key = store.keys()[0]           # Here, '/df'
        wide = store.get(key)            # Is already a pandas dataframe.
    
    wide.index.name = "ts"
    long = (
        wide.stack()
            .rename_axis(["ts", "node_id"])
            .reset_index(name="value")
    )
    value_cols = long.columns[2:]
    if len(value_cols):
        long = long.copy()
        long.loc[:, value_cols] = long.loc[:, value_cols].where(long.loc[:, value_cols] != 0)
    return long

def _individual_prepare_adj(out_dir, dataset):
    if dataset=="metrla":
        fpath=out_dir/"adj_mx.pkl"
    elif dataset=="pemsbay":
        fpath=out_dir/"adj_mx_bay.pkl"
    else:
        raise NotImplementedError("An unaccepted dataset_select value was passed to prepare(), please pass a list containing one or both of 'metrla' and 'pemsbay")
    with open(fpath, "rb") as f:
        obj = pickle.load(f, encoding="latin1")    # file object, not a string
    ids, id_to_ind, adj = obj[0], obj[1], np.asarray(obj[2])
    i, j = np.nonzero(adj); m = i != j
    edges = pd.DataFrame({"from":[ids[a] for a in i[m]],
                          "to":[ids[b] for b in j[m]],
                        "cost":adj[i[m], j[m]].astype(float)})
    return edges

def prepare(download=True, cleanup=True, out_path=OUT_DEFAULT_PATH, dataset_select: list =["pemsbay", "metrla"]):
    if download:
        print("Starting Download, this might take a while..")
        _download(dataset_select, out_dir=out_path)
        print("Download finished! Data processing started.")

    for ds in dataset_select:
        (out_path/ds).mkdir(parents=True, exist_ok=True)
        df=_individual_prepare_data(out_path, ds)
        df.to_csv(out_path/ds/"series.csv", index=False)

        df=_individual_prepare_adj(out_path, ds)
        df.to_csv(out_path/ds/"edges.csv", index=False)

    if cleanup:
        _cleanup(out_path)
    
if __name__=="__main__":
    prepare(dataset_select=["metrla"])

