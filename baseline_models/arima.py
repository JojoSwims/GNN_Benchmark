import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import SARIMAX
import util

"""
Todo:
We have to:
-I have finished ARIMA fitting, I just need to run them with MAE, RMSE, MAPE


"""


def split_into_sensor_frames(df):
    """
    Returns a dict {sensor_id: DataFrame[T x 1]} with the column named as the sensor_id.
    """
    sensor_frames = {}
    for col in df.columns:
        # Single-column DataFrame for ARIMA; keep index & freq
        sensor_frames[col] = pd.DataFrame(df[col]).copy()
    return sensor_frames

def get_mask(df, verbose: bool=True):
    """
    Returns a boolean mask (same shape as df) where True = ground truth exists.
    
    Parameters
    ----------
    df : pd.DataFrame
        Shape (T, N), indexed by time, zeros mean missing.

    Returns
    -------
    pd.DataFrame
        Boolean mask with True for valid observations.
    """
    mask=df!=0
    if verbose:
        total_vals = df.size
        missing_vals = (~mask).sum().sum()
        overall_pct = 100 * missing_vals / total_vals
        print(f"Overall missing: {overall_pct:.2f}% ({missing_vals}/{total_vals})")
        
        per_sensor_pct = (~mask).sum() / len(df) * 100
        print("\nMissing per sensor (%):")
        print(per_sensor_pct.sort_values(ascending=False))
    
    return mask



def arima_forecast(df, order=(3,0,1), seasonal_order=(1,0,0,12), train_ratio=0.8):
    """
    Runs arima (fits and then predicts) on a single time-series. 
    Returns a dict that has the resulting prediction with h as key values and the y_test with key value 0.
    NOTE: We will be missing the first h-1 observations, which we will 
    """
    HORIZONS=(3,6,12)
    y=df.iloc[:, 0] #Ensure we have a series 
    y_train, _, y_test=util.time_splits(y, train_frac=train_ratio, val_frac=0)
    res=res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit()


    #Now, fill in a list of predictions for each h
    predictions={}
    for h in HORIZONS:
        res_h=res.apply(endog=y_train) #This is basically a .copy()
        results=[]
        max_index=len(y_test)-h+1 #Max index: The index at which we ask our last prediction
        print(h)
        for i in range(max_index):
            forecast=res_h.forecast(steps=h)
            res_h=res_h.extend([y_test.iloc[i]])
            results.append(float(forecast.iloc[h-1]))
        
        pred_index = y_test.index[h-1:]
        predictions[h] = pd.Series(results, index=pred_index)

    return predictions





if __name__=="__main__":
    
    #New workflow start
    path="../temp/aqi"
    

    df=util.wide2long(path)
    #Run SARIMA on that DF and compute the error statistics with the functions in baseline_models.util
    
    

    #----------PREVIOUS WORKFLOW------------------

    df_baseline = util.load_csv("./pemsbay/pemsbay.csv").replace(0.0, np.nan)
    df_test     = util.load_csv("pemsh12.csv")

    # 1) Align on common index/columns (df_test starts later—this handles it)
    common_cols = df_baseline.columns.intersection(df_test.columns)
    common_idx  = df_baseline.index.intersection(df_test.index)

    y_true = df_baseline.loc[common_idx, common_cols]
    y_pred = df_test.loc[common_idx, common_cols]

    # 2) Score only where both are present
    mask    = y_true.notna() & y_pred.notna()
    y_trueM = y_true.where(mask)
    y_predM = y_pred.where(mask)

    # 3) Errors and metrics
    abs_err     = (y_predM - y_trueM).abs()
    squared_err = (y_predM - y_trueM)**2

    # For MAPE, ignore spots where true is 0 or NaN
    denom = y_trueM.abs().replace(0, np.nan)
    mape_mat = (abs_err / denom) * 100.0

    # Per-column metrics (sensor-wise)
    mae_per_col  = abs_err.mean(axis=0)
    rmse_per_col = np.sqrt(squared_err.mean(axis=0))
    mape_per_col = mape_mat.mean(axis=0)

    metrics_by_col = pd.DataFrame({
        "MAE": mae_per_col,
        "RMSE": rmse_per_col,
        "MAPE_%": mape_per_col
    })

    # Overall metrics (across all sensors & timestamps)
    mae_overall  = np.nanmean(abs_err.to_numpy())
    rmse_overall = np.sqrt(np.nanmean(squared_err.to_numpy()))
    mape_overall = np.nanmean(mape_mat.to_numpy())

    print("Overall metrics on overlapping, non-missing cells:")
    print(f"  MAE   : {mae_overall:.4f}")
    print(f"  RMSE  : {rmse_overall:.4f}")
    print(f"  MAPE% : {mape_overall:.2f}")

    """
    horizons = [3, 6, 12]
    order = (3, 1, 1)
    train_ratio=0.8
    trend = "c"



    df=util.load_csv("./pemsbay/pemsbay.csv")
    #Getting all the results:
    sensors=split_into_sensor_frames(df)
    
    overall_results={}
    for k in sensors:
        print(k)
        result=arima_forecast(sensors[k], order=order, train_ratio=train_ratio)
        overall_results[k]=result
    
    pred_df_h3  = pd.DataFrame({k: overall_results[k][3]  for k in sensors})
    pred_df_h6  = pd.DataFrame({k: overall_results[k][6]  for k in sensors})
    pred_df_h12 = pd.DataFrame({k: overall_results[k][12] for k in sensors})

    pred_df_h3.to_csv("pemsh3.csv")
    pred_df_h6.to_csv("pemsh6.csv")
    pred_df_h12.to_csv("pemsh12.csv")
    """