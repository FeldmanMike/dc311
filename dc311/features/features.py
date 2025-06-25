"""
Create features for model
"""

import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


logger = logging.getLogger(__name__)


def create_year_feature(series: pd.Series):
    """
    Year that 311 request was made.

    Args:
        series: A pandas Series to be transformed

    Returns:
        pandas DataFrame with new column for year
    """
    return pd.DataFrame(series.dt.year)


def create_month_feature(series: pd.Series):
    """
    Month that 311 request was made.

    Args:
        series: A pandas Series to be transformed

    Returns:
        pandas DataFrame with new column for month
    """
    return pd.DataFrame(series.dt.month)


def create_quarter_feature(series: pd.Series):
    """
    Quarter that 311 request was made.

    Args:
        series: A pandas Series to be transformed

    Returns:
        pandas DataFrame with new column for quarter
    """
    return pd.DataFrame(series.dt.quarter)


def create_day_feature(series: pd.Series):
    """
    Day of week that 311 request was made.

    Args:
        series: A pandas Series to be transformed

    Returns:
        pandas DataFrame with new column for day of week
    """
    return pd.DataFrame(series.dt.dayofweek)


def is_business_hours(hour):
    return 1 if 8 <= hour < 17 else 0


def is_business_day(day):
    return 1 if 0 <= day < 5 else 0


def create_business_hours_feature(adddate_series):
    """
    Whether 311 request was made during business hours.

    Args:
        adddate_series: A pandas Series with the `adddate` field

    Returns:
        pandas DataFrame with new column for whether request was made during business
        hours
    """
    df = pd.DataFrame(adddate_series, columns=["adddate"])
    df["add_during_business_hours"] = df["adddate"].dt.hour.apply(is_business_hours)
    df["add_during_business_day"] = df["adddate"].dt.dayofweek.apply(is_business_day)
    df["add_during_business_hours"] = np.where(
        (df["add_during_business_hours"] == 1) & (df["add_during_business_day"] == 1),
        1,
        0,
    )
    return df[["add_during_business_hours"]]


def engineer_features():
    """
    Create a reusable pipeline that engineers features

    Args:
        None

    Returns:
        sklearn ColumnTransformer object
    """
    return Pipeline(
        [
            (
                "add_new_columns",
                ColumnTransformer(
                    [
                        (
                            "add_month",
                            FunctionTransformer(create_month_feature),
                            "adddate",
                        ),
                        (
                            "add_quarter",
                            FunctionTransformer(create_quarter_feature),
                            "adddate",
                        ),
                        ("add_day", FunctionTransformer(create_day_feature), "adddate"),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=True,
                ),
            ),
            (
                "onehotencode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def select_features(feature_list: List[str]):
    """
    Create a reusable ColumnTransformer that selects features

    Args:
        feature_list: List of features to keep.

    Returns:
        sklearn ColumnTransformer object
    """
    return ColumnTransformer(
        [("feature_selector", "passthrough", feature_list)],
        verbose_feature_names_out=False,
    )


def create_feature_engineering_pipeline(feature_list: List[str]):
    """
    Create a reusable feature engineering pipeline

    Args:
        feature_list: List of features to keep.

    Returns:
        sklearn Pipeline object
    """
    feature_selector = select_features(feature_list).set_output(transform="pandas")
    feature_transformer = engineer_features().set_output(transform="pandas")
    return Pipeline(
        [
            ("feature_selector", feature_selector),
            ("feature_engineering", feature_transformer),
        ]
    )


def get_dataset_indices(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict:
    """
    Assign indices associated with training, validation, and test sets to a
    dictionary.

    Args:
        train_df: DataFrame with training data
        validation_df: DataFrame with validation data
        test_df: DataFrame with test data

    Returns:
        Dictionary where the keys are "train", "validation", and "test", and the
        values are lists of indices associated with each dataset
    """
    data_dict = {}
    data_dict["train"] = train_df.index.tolist()
    data_dict["validation"] = validation_df.index.tolist()
    data_dict["test"] = test_df.index.tolist()
    return data_dict
