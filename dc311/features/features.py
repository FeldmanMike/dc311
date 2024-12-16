"""
Create features for model
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


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


def create_month_feature(df):
    """
    Month that 311 request was made.

    Args:
        df: pandas Dataframe with data

    Returns:
        pandas DataFrame with new column for month
    """
    df["add_month"] = df["adddate"].dt.month
    return df[["add_month"]]


def create_quarter_feature(df):
    """
    Quarter that 311 request was made.

    Args:
        df: pandas Dataframe with data

    Returns:
        pandas DataFrame with new column for quarter
    """
    df["add_quarter"] = df["adddate"].dt.quarter
    return df[["add_quarter"]]


def create_day_feature(df):
    """
    Day of week that 311 request was made.

    Args:
        df: pandas Dataframe with data

    Returns:
        pandas DataFrame with new column for day of week
    """
    df["add_day"] = df["adddate"].dt.dayofweek
    return df[["add_day"]]


def is_business_hours(hour):
    return 1 if 8 <= hour < 17 else 0


def create_business_hours_feature(df):
    """
    Whether 311 request was made during business hours.

    Args:
        df: pandas Dataframe with data

    Returns:
        pandas DataFrame with new column for whether request was made during business
        hours
    """
    df["add_during_business_hours"] = df["adddate"].dt.hour.apply(is_business_hours)
    return df[["add_during_business_hours"]]


def engineer_features():
    """
    Create a reusable ColumnTransformer that engineers features

    Args:
        None

    Returns:
        sklearn ColumnTransformer object
    """
    return ColumnTransformer(
        [
            ("year_transformer", FunctionTransformer(create_year_feature), None),
            ("month_transformer", FunctionTransformer(create_month_feature), None),
            ("quarter_transformer", FunctionTransformer(create_quarter_feature), None),
            ("day_transformer", FunctionTransformer(create_day_feature), None),
            (
                "business_hour_transformer",
                FunctionTransformer(create_business_hours_feature),
                None,
            ),
            ("onehotencode", OneHotEncoder(handle_unknown="ignore"), None),
        ],
        remainder="passthrough",
    )


def select_features(feature_list: List[str]):
    """
    Create a reusable ColumnTransformer that selects features

    Args:
        feature_list: List of features to keep.

    Returns:
        sklearn ColumnTransformer object
    """
    return ColumnTransformer([("feature_selector", "passthrough", feature_list)])


def create_feature_engineering_pipeline(feature_list: List[str]):
    """
    Create a reusable feature engineering pipeline

    Args:
        feature_list: List of features to keep.

    Returns:
        sklearn Pipeline object
    """
    feature_transformer = engineer_features()
    feature_selector = select_features()
    return Pipeline(
        [
            ("feature_engineering", feature_transformer),
            ("feature_selector", feature_selector),
        ]
    )
