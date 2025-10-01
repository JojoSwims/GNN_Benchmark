import util

import numpy as np
import pandas as pd


#This entire file needs to be redone.
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

"""
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
"""

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

