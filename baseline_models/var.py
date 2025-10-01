import util
from statsmodels.tsa.statespace.structural import UnobservedComponents
import numpy as np
import pandas as pd

def fill_missing(train, test, method="kalman"):
    """
    Takes in train and test dataframes (that are assume to follow each other chronologically)
    with NaN missing values, and uses a kalman filter to fill in the missing values.
    """
    if method=="kalman":
        mask=pd.concat([train, test]).isna()
        imputed_train=train.copy()
        imputed_test=test.copy()
        for i in range(train.shape[1]):
            print(i)
            y_train=train.iloc[:,i]
            y_test=test.iloc[:,i]
            model = UnobservedComponents(
                endog=y_train,
                level='llevel',
                freq_seasonal=[
                    {'period': 288, 'harmonics': 5, 'stochastic': False},   # daily cycle
                    {'period': 2016, 'harmonics': 1, 'stochastic': False}   # weekly cycle
                ]
            )
            #We fit the above model on the train data.
            results=model.fit()
            predicted_mean = results.predict(start=y_train.index[0], end=y_train.index[-1], dynamic=False)
            imputed_train.iloc[:, i] =y_train.fillna(predicted_mean)
            #We now work on removing the NaN values on the test set:
            appended = results.append(endog=y_test, refit=False)
            pred_te  = appended.predict(start=y_test.index[0], end=y_test.index[-1], dynamic=True)
            imputed_test.iloc[:, i] = y_test.fillna(pred_te)
        return imputed_train, imputed_test, mask
    else:
        raise NotImplementedError(method + " is not an implemented method for filling missing values")

def var(train, test):
    #TODO: Need to figure out how we are doing this, ie are we doing regular VAR (apparently bad in our situation)
    #or are we running ridge/lasso (kind of shit in our mid computer.)

    #Standardize the data to make it work better with VAR
    mu = train.mean(axis=0)
    sigma = train.std(axis=0).replace(0, 1)  # avoid division by zero in case all the values are identical
    train_std=(train - mu) / sigma
    test_std=(test - mu) / sigma

    #Fit var and predict


    #Return the predicted matrix


# The below routine was used to generate a version of METRLA with filled missing values
def remove_missing_metrla():
    df=util.load_csv("./Metrla/metrla_with_missing.csv")
    df = df.replace(0.0, np.nan)
    #Split the dataset into train and test:
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    #Fill in missing values
    train, test, mask=fill_missing(train, test)
    #For testing:
    new_df=pd.concat([train, test])
    new_df.to_csv("./Metrla/no_missing_metrla.csv")

if __name__=="__main__":
    remove_missing_metrla()
    """
    old_df=util.load_csv("./Metrla/metrla.csv")
    mask=old_df.isna()
    mask.to_csv("./Metrla/missing_values_mask.csv")
    #Do what is above this line first
    df=util.load_csv("./Metrla/no_missing_metrla.csv")
    mask=util.load_csv("./Metrla/missing_values_mask.csv")
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    """

