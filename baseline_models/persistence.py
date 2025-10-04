import util
import pandas as pd
import numpy as np
def persistence_fill(y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Forward-fill the series for use as the predictor's context (past-only).
    Supports datasets where missingness is encoded as NaN.
    Returns:
      y_ffill  : forward-filled series (leading missing remain NaN)
      obs_mask : boolean Series where the ORIGINAL y was observed (non-missing)
    """
    y = y.copy()
    obs_mask = y.notna()
    y_ffill = y.ffill()  # leading NaNs stay NaN

    return y_ffill, obs_mask



def persistence_forecast(df, horizons=(1, 3, 6, 12), train_ratio=0.8):
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
    y = df.iloc[:, 1].squeeze()


    _, _, y_test = util.time_splits(y, train_frac=train_ratio, val_frac=0)
    start_idx = len(y) - len(y_test)
    predictions = {0: y_test.copy()}

    if y_test.isna().any() or pd.isna(y.iloc[start_idx - 1]):
        raise ValueError("persistence_forecast expects no missing values at indices in and just beforey_test; run your fill step first.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")

    for h in horizons:  # keep caller's order; no sorting/dedup
        n_targets = len(y_test) - h + 1

        vals = [y.iloc[start_idx - 1 + i] for i in range(n_targets)]
        idx = y_test.index[h - 1 : h - 1 + n_targets]
        predictions[h] = pd.Series(vals, index=idx)
    return predictions




if __name__=="__main__":
  p="aqi"
  p="../temp/"+p
  df_list=util.wide2long(p)

  df=df_list[0]
  print(df)
  df, mask=persistence_fill(df)
  preds=persistence_forecast(df)
  y_truth=preds[0]
  del preds[0]
  metrics={}
  for h in preds:
    if h==0:
       continue
    y_true_aligned = y_true_aligned = y_truth.iloc[h-1:]
    
    rmse = util.rmse(y_true_aligned.to_numpy(), preds[h].to_numpy())
    mape = util.mape(y_true_aligned.to_numpy(), preds[h].to_numpy())
    mae = util.mae(y_true_aligned.to_numpy(), preds[h].to_numpy())
    new_metrics = {
      "MAE"+str(h): mae,
      "RMSE"+str(h): rmse,
      "MAPE"+str(h): mape
    }
    metrics=metrics|new_metrics
  print(metrics)
  with open(p+"1.txt", "w") as f:
    for name, value in metrics.items():
      f.write(f"{name}: {value:.4f}\n")
