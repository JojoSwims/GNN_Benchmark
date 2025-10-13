from pathlib import Path
import urllib.request
import zipfile
import shutil
import pandas as pd



"""
Dataset description:
See https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 (also includes the paper to cite.)
"""
ELERGONE_NODE_ORDER=["MT_124", "MT_131", "MT_132", "MT_156", "MT_158", "MT_159", "MT_161", "MT_162", "MT_163", "MT_166", "MT_168", "MT_169", "MT_171", "MT_172", "MT_174", "MT_175", "MT_176", "MT_180", "MT_182", "MT_183", "MT_187", "MT_188", "MT_189", "MT_190", "MT_191", "MT_192", "MT_193", "MT_194", "MT_195", "MT_196", "MT_197", "MT_198", "MT_199", "MT_200", "MT_201", "MT_202", "MT_203", "MT_204", "MT_205", "MT_206", "MT_207", "MT_208", "MT_209", "MT_210", "MT_211", "MT_212", "MT_213", "MT_214", "MT_215", "MT_216", "MT_217", "MT_218", "MT_219", "MT_220", "MT_221", "MT_222", "MT_223", "MT_225", "MT_226", "MT_227", "MT_228", "MT_229", "MT_230", "MT_231", "MT_232", "MT_233", "MT_234", "MT_235", "MT_236", "MT_237", "MT_238", "MT_239", "MT_240", "MT_241", "MT_242", "MT_243", "MT_244", "MT_245", "MT_246", "MT_247", "MT_248", "MT_249", "MT_250", "MT_251", "MT_252", "MT_253", "MT_254", "MT_256", "MT_257", "MT_258", "MT_259", "MT_260", "MT_261", "MT_262", "MT_263", "MT_264", "MT_265", "MT_266", "MT_267", "MT_268", "MT_269", "MT_270", "MT_271", "MT_272", "MT_273", "MT_274", "MT_275", "MT_276", "MT_277", "MT_278", "MT_279", "MT_281", "MT_282", "MT_283", "MT_284", "MT_285", "MT_286", "MT_287", "MT_288", "MT_290", "MT_291", "MT_292", "MT_293", "MT_294", "MT_295", "MT_296", "MT_297", "MT_298", "MT_299", "MT_300", "MT_301", "MT_302", "MT_303", "MT_304", "MT_306", "MT_307", "MT_309", "MT_310", "MT_311", "MT_312", "MT_313", "MT_314", "MT_315", "MT_316", "MT_317", "MT_318", "MT_319", "MT_320", "MT_321", "MT_323", "MT_324", "MT_325", "MT_326", "MT_327", "MT_328", "MT_329", "MT_330", "MT_331", "MT_164", "MT_280", "MT_001", "MT_002", "MT_003", "MT_004", "MT_005", "MT_006", "MT_007", "MT_008", "MT_009", "MT_010", "MT_011", "MT_013", "MT_014", "MT_016", "MT_017", "MT_018", "MT_019", "MT_020", "MT_021", "MT_022", "MT_023", "MT_025", "MT_026", "MT_027", "MT_028", "MT_029", "MT_031", "MT_034", "MT_035", "MT_036", "MT_037", "MT_038", "MT_040", "MT_042", "MT_043", "MT_044", "MT_045", "MT_046", "MT_047", "MT_048", "MT_049", "MT_050", "MT_051", "MT_052", "MT_053", "MT_054", "MT_055", "MT_056", "MT_057", "MT_058", "MT_059", "MT_060", "MT_061", "MT_062", "MT_063", "MT_064", "MT_065", "MT_066", "MT_067", "MT_068", "MT_069", "MT_070", "MT_071", "MT_072", "MT_073", "MT_074", "MT_075", "MT_076", "MT_077", "MT_078", "MT_079", "MT_080", "MT_081", "MT_082", "MT_083", "MT_084", "MT_085", "MT_086", "MT_087", "MT_088", "MT_089", "MT_090", "MT_091", "MT_093", "MT_094", "MT_095", "MT_096", "MT_097", "MT_098", "MT_099", "MT_100", "MT_101", "MT_102", "MT_103", "MT_104", "MT_105", "MT_114", "MT_118", "MT_119", "MT_123", "MT_125", "MT_126", "MT_128", "MT_129", "MT_130", "MT_135", "MT_136", "MT_137", "MT_138", "MT_139", "MT_140", "MT_141", "MT_142", "MT_143", "MT_145", "MT_147", "MT_148", "MT_149", "MT_150", "MT_151", "MT_153", "MT_154", "MT_155", "MT_157", "MT_333", "MT_334", "MT_335", "MT_336", "MT_338", "MT_339", "MT_340", "MT_341", "MT_342", "MT_343", "MT_344", "MT_345", "MT_346", "MT_347", "MT_348", "MT_349", "MT_350", "MT_351", "MT_352", "MT_353", "MT_354", "MT_355", "MT_356", "MT_357", "MT_358", "MT_359", "MT_360", "MT_361", "MT_362", "MT_363", "MT_364", "MT_365", "MT_366", "MT_367", "MT_368", "MT_369", "MT_146", "MT_173", "MT_024", "MT_167", "MT_127", "MT_033", "MT_177", "MT_184", "MT_308", "MT_332", "MT_144", "MT_255", "MT_032", "MT_134", "MT_152", "MT_370", "MT_092", "MT_170", "MT_030", "MT_015", "MT_185", "MT_224", "MT_012", "MT_322", "MT_165", "MT_041", "MT_186", "MT_039", "MT_289", "MT_305", "MT_179", "MT_106", "MT_107", "MT_108", "MT_110", "MT_111", "MT_113", "MT_115", "MT_117", "MT_120", "MT_121", "MT_122", "MT_337", "MT_160", "MT_112", "MT_109", "MT_116", "MT_181", "MT_133", "MT_178"]


URL="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/elergone")
DOWNLOAD_NAME="LD2011_2014.txt"

def _download(zip_path = ZIP_DEFAULT_PATH, out_dir = OUT_DEFAULT_PATH):
#Download:
    urllib.request.urlretrieve(URL, str(zip_path))

    #Unzip:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        print("Unzipped to", out_dir)
    #Remove unnecessary downloads
    try:
        shutil.rmtree(out_dir/"__MACOSX")
    except FileNotFoundError:
        pass

    #Remove the zip file after extraction
    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass



def prepare(download=True, cleanup=True, out_path=OUT_DEFAULT_PATH):
    if download:
        print("Downloading data...")
        _download(out_dir=out_path)
    print("Preparing the data, this will take a while.")
    df=pd.read_csv(
        out_path / DOWNLOAD_NAME,
        sep=";",                 # semicolon delimiter
        quotechar='"',           # quoted fields
        decimal=",",             # comma decimals
        parse_dates=[0],         # first column is datetime
        index_col=0,             # make it the index (so math is clean)
    )
    df.index.name = "timestamp"

    #Dividing by 4 makes the value go from kW to kWh (described in the dataset description above)
    df=df/4.0

    # Replace zero measurements with NaN before reshaping the dataset
    if not df.empty:
        value_columns = df.columns.tolist()
        df.loc[:, value_columns] = df.loc[:, value_columns].where(df.loc[:, value_columns] != 0)

    #Pivot the dataframe to our intermediate representation:
    df = (
        df.stack()
          .rename_axis(["ts", "node_id"])
          .reset_index(name="value")
          .sort_values("ts", kind="mergesort")
          .reset_index(drop=True)
    )

    #Output our intermediate representation:
    df.to_csv(out_path/"series.csv", index=False)

    #Output our mask:
    id_cols   = df.columns[:2]
    feat_cols = df.columns[2]
    observation_mask = df[id_cols].join(df[feat_cols].notna())
    observation_mask.to_csv(out_path/"mask.csv", index=False)
    
    if cleanup:
        try:
            (out_path / DOWNLOAD_NAME).unlink()
        except FileNotFoundError:
            pass

if __name__=="__main__":
    prepare(download=True)