import pandas as pd
from statsmodels.tsa.arima.model import SARIMAX, UnobservedComponents
import numpy as np

import util


def kalman_impute(df: pd.DataFrame, minutes_in_step: int = 5, train_ratio: float = 0.8):
    """
    Impute NaNs column-wise using a UnobservedComponents (local level + deterministic daily/weekly seasonality).
    Fits parameters on the first `train_ratio` fraction (train) and applies to the rest (test) without refitting.
    Returns (imputed_df, missing_mask).
    """
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
    mask    = df.isna()

    for i in range(df.shape[1]):
        y_tr = pd.to_numeric(df.iloc[:n_train, i], errors="coerce")
        y_te = pd.to_numeric(df.iloc[n_train:, i], errors="coerce")

        # if train has no finite values, skip this column
        if not np.isfinite(y_tr).any():
            continue

        model = UnobservedComponents(
            endog=y_tr,
            level="llevel",
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal
        )
        res = model.fit(disp=False)

        # in-sample predictions for train, condition on observed points
        pred_tr = res.predict(start=idx_tr[0], end=idx_tr[-1], dynamic=False)
        imputed.iloc[:n_train, i] = y_tr.fillna(pred_tr)

        # apply fixed parameters to test without refit; condition on observed test points
        if len(idx_te) > 0:
            appended = res.append(endog=y_te, refit=False)
            pred_te = appended.predict(start=idx_te[0], end=idx_te[-1], dynamic=False)
            imputed.iloc[n_train:, i] = y_te.fillna(pred_te)

    return imputed, mask

def split_into_sensor_frames(df):
    """
    Returns a dict {sensor_id: DataFrame[T x 1]} with the column named as the sensor_id.
    """
    sensor_frames = {}
    for col in df.columns:
        # Single-column DataFrame for ARIMA; keep index & freq
        sensor_frames[col] = pd.DataFrame(df[col]).copy()
    return sensor_frames

def sarima_forecast(df, order=(3, 0, 1), seasonal_order=(1, 0, 0, 12), train_ratio=0.8):
    """

    """
    horizons = (3, 6, 12)
    y = df.iloc[:, 0]
    y_train, _, y_test = util.time_splits(y, train_frac=train_ratio, val_frac=0)
    res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit()

    predictions = {}
    for horizon in horizons:
        if len(y_test) < horizon:
            continue

        res_h = res.apply(endog=y_train)
        preds = []
        max_index = len(y_test) - horizon + 1

        for i in range(max_index):
            forecast = res_h.forecast(steps=horizon)
            preds.append(float(forecast.iloc[horizon - 1]))
            res_h = res_h.extend([y_test.iloc[i]])

        pred_index = y_test.index[horizon - 1 :]
        aligned_truth = y_test.iloc[horizon - 1 : horizon - 1 + len(preds)]
        predictions[horizon] = pd.DataFrame(
            {"y_true": aligned_truth.to_numpy(dtype=float), "y_pred": preds},
            index=pred_index,
        )

    return predictions


def compute_metrics(prediction_df):
    """Compute MAE, RMSE and MAPE for a prediction DataFrame."""

    if prediction_df.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

    y_true = prediction_df["y_true"].to_numpy(dtype=float)
    y_pred = prediction_df["y_pred"].to_numpy(dtype=float)

    return {
        "MAE": util.mae(y_true, y_pred),
        "RMSE": util.rmse(y_true, y_pred),
        "MAPE": util.mape(y_true, y_pred),
    }


def evaluate_sensor(sensor_df, order, seasonal_order, train_ratio):
    """Run ARIMA for all horizons on a single sensor frame and compute metrics."""

    predictions = sarima_forecast(
        sensor_df, order=order, seasonal_order=seasonal_order, train_ratio=train_ratio
    )

    metrics = {}
    for horizon, pred_df in predictions.items():
        metrics[horizon] = compute_metrics(pred_df)

    return predictions, metrics


def run_pipeline(path, order=(3, 0, 1), seasonal_order=(1, 0, 0, 12), train_ratio=0.8):
    """Execute the ARIMA workflow on all datasets located at ``path``."""

    datasets = util.wide2long(path)
    all_results = []

    for dataset_idx, dataset in enumerate(datasets):
        time_col = dataset.columns[0]
        wide = dataset.set_index(time_col)
        sensors = split_into_sensor_frames(wide)

        for sensor_name, sensor_df in sensors.items():
            preds, metrics = evaluate_sensor(
                sensor_df,
                order=order,
                seasonal_order=seasonal_order,
                train_ratio=train_ratio,
            )
            all_results.append(
                {
                    "dataset": dataset_idx,
                    "sensor": sensor_name,
                    "predictions": preds,
                    "metrics": metrics,
                }
            )

    return all_results


def print_metrics(results):
    """Pretty-print the metrics for the computed forecasts."""

    for record in results:
        dataset = record["dataset"]
        sensor = record["sensor"]
        print(f"Dataset {dataset}, Sensor {sensor}")
        for horizon, values in sorted(record["metrics"].items()):
            mae = values["MAE"]
            rmse = values["RMSE"]
            mape = values["MAPE"]
            print(
                f"  Horizon {horizon}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}"
            )
        print()


if __name__ == "__main__":
    PATH = "../temp/aqi"
    ORDER = (3, 0, 1)
    #Set the seasonal order to something meaningful
    SEASONAL_ORDER = (1, 0, 0, 12)
    TRAIN_RATIO = 0.8

    results = run_pipeline(PATH, order=ORDER, seasonal_order=SEASONAL_ORDER, train_ratio=TRAIN_RATIO)
    print_metrics(results)
