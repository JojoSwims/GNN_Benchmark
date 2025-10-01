import util
import pandas as pd

"""
For historical averages, we use the code provided in the DCRNN GitHub.
TODO: Make sure this is cited!! 
"""



#------------------------------------------------------------------------------------
import numpy as np
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)
#------------------------------------------------------------------------------------




def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test

def ha_metrla():
    df=util.load_csv("./Elergone/metrla.csv")
    y_predict, y_test=historical_average_predict(df)
    
    # Assuming y_predict, y_test are DataFrames
    y_true_vals = y_test.to_numpy(dtype=float)
    y_pred_vals = y_predict.to_numpy(dtype=float)

    # Compute metrics
    metrics = {
        "MAE": util.mae(y_true_vals, y_pred_vals),
        "RMSE": util.rmse(y_true_vals, y_pred_vals),
        "MAPE": util.mape(y_true_vals, y_pred_vals)
    }
    with open("metrics_results.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")


def eval_historical_average(traffic_reading_df):
    y_predict, y_test = historical_average_predict(traffic_reading_df, test_ratio=0.2)
    rmse = masked_rmse_np(preds=y_predict.to_numpy(), labels=y_test.to_numpy(), null_val=0)
    mape = masked_mape_np(preds=y_predict.to_numpy(), labels=y_test.to_numpy(), null_val=0)
    mae = masked_mae_np(preds=y_predict.to_numpy(), labels=y_test.to_numpy(), null_val=0)
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    with open("metrics_results.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

if __name__=="__main__":
    df=util.load_csv("./BeijingAir/beijing_air_quality.csv")
    eval_historical_average(df)