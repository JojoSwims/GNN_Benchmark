import pandas as pd
from statsmodels.tsa.arima.model import SARIMAX

import util


def split_into_sensor_frames(df):
    """
    Returns a dict {sensor_id: DataFrame[T x 1]} with the column named as the sensor_id.
    """
    sensor_frames = {}
    for col in df.columns:
        # Single-column DataFrame for ARIMA; keep index & freq
        sensor_frames[col] = pd.DataFrame(df[col]).copy()
    return sensor_frames

def arima_forecast(df, order=(3, 0, 1), seasonal_order=(1, 0, 0, 12), train_ratio=0.8):
    """
    Runs arima (fits and then predicts) on a single time-series. 
    Returns a dict that has the resulting prediction with h as key values and the y_test with key value 0.
    NOTE: We will be missing the first h-1 observations, which we will 
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

    predictions = arima_forecast(
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
