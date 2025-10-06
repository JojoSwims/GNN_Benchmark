import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

import util
from util import log_status

def _minutes_to_freq(minutes: int) -> str:
    """Return a pandas offset alias like '5min' or '1H'."""
    return f"{minutes//60}H" if minutes % 60 == 0 else f"{minutes}min"

def _minutes_to_freq(minutes: int) -> str:
    """Return a pandas offset alias like '5min' or '1H'."""
    return f"{minutes//60}H" if minutes % 60 == 0 else f"{minutes}min"


def kalman_impute(df: pd.DataFrame, minutes_in_step: int = 5, train_ratio: float = 0.8):
    """
    Impute NaNs column-wise using a UnobservedComponents (local level + deterministic daily/weekly seasonality).
    Fits parameters on the first `train_ratio` fraction (train) and applies to the rest (test) without refitting.
    Returns (imputed_df, missing_mask).
    """
    log_status(
        "Starting Kalman imputation for DataFrame "
        f"with shape={df.shape} and minutes_in_step={minutes_in_step}"
    )
    freq_str = _minutes_to_freq(minutes_in_step)
    if len(df.columns) > 0 and is_datetime64_any_dtype(df.iloc[:, 0]):
        df = df.set_index(df.columns[0])
    if is_datetime64_any_dtype(df.index):
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        try:
            df.index = df.index.tz_convert(None)     # CHANGED: tz-naive if tz-aware
        except Exception:
            pass
    # periods from sampling step
    daily  = max(2, int(round(1440 / minutes_in_step)))
    weekly = max(2, daily * 7)

    freq_seasonal = [
        {"period": daily,  "harmonics": 5},  # daily cycle
        {"period": weekly, "harmonics": 1},  # weekly cycle
    ]
    stochastic_freq_seasonal = [False, False]

    n_train = int(round(len(df) * train_ratio))
    idx_tr  = df.index[:n_train]
    idx_te  = df.index[n_train:]

    imputed = df.copy()
    mask    = df.notna().astype(bool)

    for i in range(df.shape[1]):
        log_status(
            "Kalman impute column "
            f"{i + 1}/{df.shape[1]} ('{df.columns[i]}')"
        )
        y_tr = pd.to_numeric(df.iloc[:n_train, i], errors="coerce")
        y_te = pd.to_numeric(df.iloc[n_train:, i], errors="coerce")

        # if train has no finite values, skip this column
        if not np.isfinite(y_tr).any():
            log_status(
                f"Skipping column '{df.columns[i]}' due to no finite values in training"
            )
            continue

        model = UnobservedComponents(
            endog=y_tr,
            level="llevel",
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal
        )
        res = model.fit(disp=False)
        log_status(f"Model fit complete for column '{df.columns[i]}'")

        # in-sample predictions for train, condition on observed points
        pred_tr = res.predict(start=idx_tr[0], end=idx_tr[-1], dynamic=False)
        imputed.iloc[:n_train, i] = y_tr.fillna(pred_tr)

        # apply fixed parameters to test without refit; condition on observed test points
        if len(idx_te) > 0:
            appended = res.append(endog=y_te, refit=False)
            pred_te = appended.predict(start=idx_te[0], end=idx_te[-1], dynamic=False)
            imputed.iloc[n_train:, i] = y_te.fillna(pred_te)
            log_status(
                f"Imputed test segment for column '{df.columns[i]}' "
                f"with {len(idx_te)} timestamps"
            )

    log_status("Completed Kalman imputation")

    return imputed, mask

def _is_time_like_col(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    try:
        pd.to_datetime(s, errors="raise", infer_datetime_format=True)
        return True
    except Exception:
        return False


def split_into_sensor_frames(df: pd.DataFrame, mask: pd.DataFrame):
    """
    Returns dict[sensor_id] -> (one_col_df, one_col_mask), skipping col0 if time-like.
    Assumes df and mask share the same index/columns.
    """
    start_col = 1 if _is_time_like_col(df.iloc[:, 0]) else 0
    out = {}
    for j in range(start_col, df.shape[1]):
        col = df.columns[j]
        out[col] = (df.iloc[:, [j]].copy(), mask.iloc[:, [j]].copy())
    log_status(
        "Split DataFrame into sensor frames: "
        f"{len(out)} sensors detected"
    )
    return out


def sarima_forecast(df, mask, order=(3, 0, 1), seasonal_order=(1, 0, 0, 12), train_ratio=0.8):
    """
    Our Sarima forecast, for one sensor.
    """
    series_name = df.columns[0] if len(df.columns) else "<unknown>"
    log_status(
        f"Starting SARIMA forecast for series '{series_name}' with {len(df)} rows"
    )
    horizons = (1, 3, 6, 12)
    y = df.iloc[:, 0]
    y_mask=mask.iloc[:, 0]
    y_train, _, y_test = util.time_splits(y, train_frac=train_ratio, val_frac=0)
    _,_,mask_test=util.time_splits(y_mask, train_frac=train_ratio, val_frac=0)
    log_status(
        f"Series '{series_name}': train={len(y_train)}, test={len(y_test)}"
    )
    res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit()
    log_status(
        f"Fitted SARIMA model for series '{series_name}' with order={order} "
        f"and seasonal_order={seasonal_order}"
    )

    predictions = {}
    for horizon in horizons:
        if len(y_test) < horizon:
            log_status(
                f"Series '{series_name}': skipping horizon {horizon} (insufficient test samples)"
            )
            continue

        res_h = res.apply(endog=y_train)
        preds = []
        max_index = len(y_test) - horizon + 1
        log_status(
            f"Series '{series_name}': forecasting horizon {horizon} over {max_index} windows"
        )

        for i in range(max_index):
            forecast = res_h.forecast(steps=horizon)
            preds.append(float(forecast.iloc[horizon - 1]))
            res_h = res_h.extend([y_test.iloc[i]])

        pred_index = y_test.index[horizon - 1 :]
        aligned_truth = y_test.iloc[horizon - 1 : horizon - 1 + len(preds)]
        aligned_mask = mask_test.iloc[horizon - 1 : horizon - 1 + len(preds)]
        predictions[horizon] = pd.DataFrame(
            {"y_true": aligned_truth.to_numpy(dtype=float), "y_pred": preds, "mask":aligned_mask},
            index=pred_index,
        )
        log_status(
            f"Series '{series_name}': completed horizon {horizon} forecasts"
        )
    log_status(f"Finished SARIMA forecast for series '{series_name}'")
    return predictions

def batch_predict_by_sensor(df: pd.DataFrame, mask: pd.DataFrame, order, seasonal_order):
    """
    1) Split into per-sensor (1-col) frames for values and mask.
    2) Run sarima_forecast(one_col_df, one_col_mask) per sensor.
    3) Merge per-horizon outputs into DataFrames with identical column order.

    Returns:
      {
        "y_pred": {h: DataFrame},
        "y_true": {h: DataFrame},
        "y_mask": {h: DataFrame}
      }
    """
    log_status("Beginning batch SARIMA predictions by sensor")
    sensor_map = split_into_sensor_frames(df, mask)  # expects {sensor_id: (df_1col, mask_1col)}
    log_status(f"Detected {len(sensor_map)} sensors for batch prediction")

    per_h_pred, per_h_true, per_h_mask = {}, {}, {}

    for idx, (sensor_id, (df_one, mask_one)) in enumerate(sensor_map.items(), start=1):
        log_status(
            f"Processing sensor '{sensor_id}' ({idx}/{len(sensor_map)})"
        )
        out = sarima_forecast(df_one, mask_one, order, seasonal_order)  # {h: DataFrame['y_true','y_pred','mask']}

        for h, df_h in out.items():
            per_h_pred.setdefault(h, {})[sensor_id] = df_h["y_pred"]
            per_h_true.setdefault(h, {})[sensor_id] = df_h["y_true"]
            per_h_mask.setdefault(h, {})[sensor_id] = df_h["mask"]

    log_status("Completed batch SARIMA predictions; consolidating results")

    # Stitch per horizon; enforce a consistent column order
    y_pred, y_true, y_mask = {}, {}, {}
    for h in per_h_pred:
        df_pred = pd.concat(per_h_pred[h], axis=1)
        cols = sorted(df_pred.columns)  # deterministic ordering across all three
        y_pred[h] = df_pred[cols]
        y_true[h] = pd.concat(per_h_true[h], axis=1)[cols]
        y_mask[h] = pd.concat(per_h_mask[h], axis=1)[cols]

    return {"y_pred": y_pred, "y_true": y_true, "y_mask": y_mask}

def compute_errors_by_h(pred_bundle):
    """
    Compute overall MAE/RMSE/MAPE per horizon h from the dict returned by
    batch_predict_by_sensor(...).

    Parameters
    ----------
    pred_bundle : dict
        {
          "y_pred": {h: pd.DataFrame},  # index: timestamps, columns: sensors
          "y_true": {h: pd.DataFrame},
          "y_mask": {h: pd.DataFrame}   # boolean or 0/1; same shape as above
        }

    Returns
    -------
    dict[int, dict[str, float]]
        { h: {"MAE": float, "RMSE": float, "MAPE": float} }
    """
    y_pred_dict = pred_bundle["y_pred"]
    y_true_dict = pred_bundle["y_true"]
    y_mask_dict = pred_bundle["y_mask"]

    results = {}
    # Iterate over horizons that exist in predictions
    for h in y_pred_dict:
        y_pred_df = y_pred_dict[h]
        y_true_df = y_true_dict[h]
        mask_df   = y_mask_dict.get(h, None)

        # Flatten all sensors/timestamps into arrays
        y_pred = y_pred_df.to_numpy(dtype=float)
        y_true = y_true_df.to_numpy(dtype=float)
        mask   = None if mask_df is None else mask_df.to_numpy().astype(bool)

        results[h] = {
            "MAE":  util.mae(y_true, y_pred, mask=mask),
            "RMSE": util.rmse(y_true, y_pred, mask=mask),
            "MAPE": util.mape(y_true, y_pred, mask=mask),
        }
        log_status(
            f"Horizon {h}: MAE={results[h]['MAE']}, "
            f"RMSE={results[h]['RMSE']}, MAPE={results[h]['MAPE']}"
        )

    return results

def write_errors_txt(results, path):
    """
    results: {h: {"MAE": float, "RMSE": float, "MAPE": float}, ...}
    path: output file path, e.g. "errors.txt"
    """
    log_status(f"Writing error metrics to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for h in sorted(results):
            f.write(f"Horizon {h}\n")
            for name, value in results[h].items():
                f.write(f"  {name}: {value:.4f}\n")
            f.write("\n")
    log_status(f"Finished writing error metrics to {path}")

def from_name(name, index, order, seasonal_order, steps):
    log_status(
        f"Starting pipeline for dataset '{name}' (index={index}, steps={steps})"
    )
    p=name
    p="../temp/"+p
    df_list=util.wide2long(p)
    df=df_list[index]
    log_status(
        f"Selected DataFrame at index {index} for dataset '{name}' with shape {df.shape}"
    )
    df_full, mask=kalman_impute(df, minutes_in_step=steps)
    preds=batch_predict_by_sensor(df_full, mask, order, seasonal_order)
    res=compute_errors_by_h(preds)
    write_errors_txt(res, p+".txt")
    log_status(f"Completed pipeline for dataset '{name}'")

def do_all():
    log_status("Starting SARIMA processing for all configured datasets")
    from_name("metrla", 0, order=(1,0,1), seasonal_order=(1,1,1,288), steps=5)
    from_name("pemsbay", 0, order=(2,0,1), seasonal_order=(1,1,1,288), steps=5)
    from_name("aqi", 0, order=(1,0,1), seasonal_order=(1,1,1,24), steps=60)
    from_name("elergone", 0, order=(1,1,1), seasonal_order=(1,1,1,96), steps=15)
    from_name("PEMS04", 2, order=(1,0,1), seasonal_order=(0,1,1,288), steps=15)
    from_name("PEMS08", 2, order=(2,0,2), seasonal_order=(0,1,1,288), steps=15)
    log_status("Completed SARIMA processing for all datasets")


if __name__ == "__main__":


    do_all()

    """
    df=pd.read_csv("test.csv")
    df_full, mask=kalman_impute(df, minutes_in_step=5)
    print(mask)
    preds=batch_predict_by_sensor(df_full, mask, order=(3,0,1), seasonal_order=(1, 0, 0, 6))
    res=compute_errors_by_h(preds)
    write_errors_txt(res, "res.txt")
    """
