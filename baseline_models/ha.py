import util
import pandas as pd

"""
For historical averages, we use the code provided in the DCRNN GitHub.
TODO: Make sure this is cited!! 
"""

def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.2):
    """
    Calculates the historical average of sensor reading. 
    This function can be provided with dataframes with missing values.
    :param df: Takes a long form df??
    :param period: number of timesteps, default is one week.
    :param test_ratio: 
    :param null_val: default 0.
    :return: y_predict, y_test
    """
    
    df = df.copy()
    df = df.fillna(0)

    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != 0.0].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test



def eval_historical_average(traffic_reading_df):
    y_predict, y_test = historical_average_predict(traffic_reading_df, test_ratio=0.2)
    rmse = util.rmse(preds=y_predict.to_numpy(), labels=y_test.to_numpy())
    mape = util.mape(preds=y_predict.to_numpy(), labels=y_test.to_numpy())
    mae = util.masked_mae_np(preds=y_predict.to_numpy(), labels=y_test.to_numpy())
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    with open("metrics_results.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")




if __name__=="__main__":

    #Our historical average runs the following way:
    paths=["../temp/aqi"] #TODO, add the paths to the 
    
    #Todo, finish writing this script, output to file and everything.
    for p in paths:
        df_list=util.wide2long(p)
        for df in df_list:
            #TODO: Change the output of the below so that stuff actually works
            eval_historical_average(df)
        if len(df_list)>1:
            #TODO: Merge the error metric (or not??)
            pass

    
