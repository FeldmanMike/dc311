"""
Test dc311/features/features.py
"""

import logging

import pandas as pd

import dc311.features.features as feat


def test_create_year_feature(time_dataframe):
    time_dataframe["add_year"] = feat.create_year_feature(time_dataframe["adddate"])
    assert time_dataframe["add_year"].to_list() == time_dataframe["year"].to_list()


def test_create_month_feature(time_dataframe):
    time_dataframe["add_month"] = feat.create_month_feature(time_dataframe["adddate"])
    assert time_dataframe["add_month"].to_list() == time_dataframe["month"].to_list()


def test_create_quarter_feature(time_dataframe):
    time_dataframe["add_quarter"] = feat.create_quarter_feature(
        time_dataframe["adddate"]
    )
    assert time_dataframe["add_quarter"].to_list() == [1, 2, 3]


def test_create_day_feature(time_dataframe):
    time_dataframe["add_day"] = feat.create_day_feature(time_dataframe["adddate"])
    assert time_dataframe["add_day"].to_list() == [6, 4, 3]


def test_is_business_hours(time_dataframe):
    time_dataframe["test"] = time_dataframe["adddate"].dt.hour.apply(
        feat.is_business_hours
    )
    assert time_dataframe["test"].to_list() == [0, 1, 0]


def test_is_business_day(time_dataframe):
    time_dataframe["test"] = time_dataframe["adddate"].dt.dayofweek.apply(
        feat.is_business_day
    )
    assert time_dataframe["test"].to_list() == [0, 1, 1]


def test_create_business_hours_feature(time_dataframe):
    return_df = feat.create_business_hours_feature(time_dataframe)
    assert return_df["add_during_business_hours"].to_list() == [0, 1, 0]


def test_engineer_features(time_dataframe):
    test_df = time_dataframe.drop(
        columns=["year", "month", "day", "hour", "minute", "second"]
    )
    feature_list = [
        "add_year",
        "add_month",
        "add_quarter",
        "add_day",
        "add_during_business_hours",
    ]
    feature_tform = feat.engineer_features(feature_list).set_output(transform="pandas")
    feature_df = feature_tform.fit_transform(test_df)
    logging.info(f"feature_df.columns: {feature_df.columns}")
    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_df) == 3
