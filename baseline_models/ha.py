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


    #Ensures we have no problems regarding misaligned timestamps or missing timestamps
    u = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])).unique().sort_values()
    f = pd.infer_freq(u) or u.to_series().diff().mode().iloc[0]
    assert u.equals(pd.date_range(u[0], u[-1], freq=f))

    
    df = df.copy()
    df = df.set_index(df.columns[0])

    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical.mean(skipna=True)
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test



def eval_historical_average(df, period):
    y_predict, y_test = historical_average_predict(df, period=period, test_ratio=0.2)
    rmse = util.rmse(y_test.to_numpy(), y_predict.to_numpy())
    mape = util.mape(y_test.to_numpy(), y_predict.to_numpy())
    mae = util.mae(y_test.to_numpy(), y_predict.to_numpy())


    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    return metrics





if __name__=="__main__":


    paths=["aqi",]
    for p in paths:
        fname=p
        p="../temp/"+p
        df_list=util.wide2long(p)

        df=df_list[0]
        zero_count = df.eq(0).sum().sum()

        res=eval_historical_average(df, period=24 * 7)
        print(res)
        with open(p+"2.txt", "w") as f:
            for name, value in res.items():
                f.write(f"{name}: {value:.4f}\n")

    
