import util
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

#Y is target, X is covariate.

alphas = 10.0 ** np.linspace(-4, 3, 20)
cv = KFold(n_splits=5, shuffle=False)  # for time series, prefer TimeSeriesSplit or custom blocks
model_cv = RidgeCV(alphas=alphas, cv=cv).fit(X_train, Y_train)
best_alpha = model_cv.alpha_

if __name__=="__main__":
    pass

