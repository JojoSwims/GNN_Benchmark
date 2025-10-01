import util
import pandas as pd
import numpy as np
def persistence_fill(y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Forward-fill the series for use as the predictor's context (past-only).
    Supports datasets where missingness is encoded EITHER as NaN OR as 0 (not both).
    Returns:
      y_ffill  : forward-filled series (leading missing remain NaN)
      obs_mask : boolean Series where the ORIGINAL y was observed (non-missing)
    """
    y = y.copy()
    obs_mask = y.notna()
    y_ffill = y.ffill()  # leading NaNs stay NaN
    
    return y_ffill, obs_mask



def persistence_forecast(df, HORIZONS=(3, 6, 12), train_ratio=0.8):
    """
    Persistence baseline (naive): y_hat(t+h | info up to t-1) = y(t-1).
    Mirrors the alignment in your ARIMA loop:
      - at test step i, you forecast first (using history up to t-1),
        then you append y_test[i] to the history for the next step.
      - therefore targets are aligned to y_test.index[h-1:].
    
    Returns:
      dict with:
        - predictions[h]: pd.Series of length len(y_test)-h+1, index = y_test.index[h-1:]
        - predictions[0]: full y_test segment (for convenience)
    """
    y = df.iloc[:, 0].squeeze()                 # 1-D Series
    y_train, _, y_test = util.time_splits(y, train_frac=train_ratio, val_frac=0)

    #if y_train.zeroes to NaNs has any NaN values, throw an error:




if __name__=="__main__":
    df=pd.DataFrame() #Placeholder value, empty dataframe.
    df, mask=persistence_fill
