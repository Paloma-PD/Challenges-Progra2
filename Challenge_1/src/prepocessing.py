# Script for loading and preprocessing data
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Load dataframe
def load_data_frame(path=None, num_samples:Optional[int]=None, random_seed: int = 42) -> pd.DataFrame:
    """"
    : param path: path of the csv file.
    : param num_samples: the number of samples to draw from the data frame; if None, use all samples.
    : param random_seed: the random seed to use when sampling data points
    """

    df = pd.read_csv(filepath_or_buffer=path)

    if num_samples is not None:
        df = df.sample(num_samples,
                       random_state=random_seed)
    print("Data is loaded")
    
    return df

# Prepocessing
def preprocessing_data(df,scaling:bool=False):
    """
    :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
    """

    # Since there are two columns that do not provide us with information, we will eliminate them.
    df = df.drop(columns=["id", "Unnamed: 32"])

    # Encode the target variable (M = 1, B = 0)
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])

    # The characteristics (independent variables) and the target variable are defined
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    if scaling:
        # Scaling data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Data has been scaled: StandardScaler")

    return X, y