"""
Create features for model
"""

import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def create_year_feature(df):
    """
    Year that 311 request was made.

    Args:
        df: pandas Dataframe with data

    Returns:
        pandas DataFrame with new column for year
    """
    df["add_year"] = df["adddate"].dt.year
    return df[["add_year"]]


# TODO: Add other functions for feature engineering
# create function transformers for this functions
# pass all function transformers to a column transformer
# include steps in column transformer at end for selecting features and OneHotEncode
# (at end)
