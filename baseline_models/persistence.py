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



def persistence_forecast(df, horizons=(3, 6, 12), train_ratio=0.8):
    """
    Persistence baseline (naive): y_hat(t+h | info up to t-1) = y(t-1).
      - at test step i, you forecast first (using history up to t-1),
        then you append y_test[i] to the history for the next step.
      - therefore targets are aligned to y_test.index[h-1:].
    
    Returns:
      dict with:
        - predictions[h]: pd.Series of length len(y_test)-h+1, index = y_test.index[h-1:]
        - predictions[0]: full y_test segment (for convenience)
    """
    y = df.iloc[:, 0].squeeze()                 # 1-D Series

    if y.isna().any():
        raise ValueError("persistence_forecast expects no missing values; run your fill step first.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    
    _, _, y_test = util.time_splits(y, train_frac=train_ratio, val_frac=0)

    start_idx = y.index.get_loc(y_test.index[0])
    predictions = {0: y_test.copy()}
    for h in horizons:  # keep caller's order; no sorting/dedup
        n_targets = len(y_test) - h + 1
        n_targets = len(y_test) - h + 1

        vals = [y.iloc[start_idx - 1 + i] for i in range(n_targets)]
        idx = y_test.index[h - 1 : h - 1 + n_targets]
        predictions[h] = pd.Series(vals, index=idx)
    return predictions




if __name__=="__main__":
    df=pd.DataFrame() #Placeholder value, empty dataframe.
    df, mask=persistence_fill(df)
    preds=persistence_forecast(df)
    #Compute the error rates:
