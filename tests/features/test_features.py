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
    feature_tform = feat.engineer_features().set_output(transform="pandas")
    feature_df = feature_tform.fit_transform(test_df)
    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_df) == 3
    assert feature_df.shape[1] == 11
    assert feature_df.isin([0, 1]).all().all()


def test_select_features(time_dataframe):
    test_df = time_dataframe.drop(
        columns=["year", "month", "day", "hour", "minute", "second"]
    )
    test_df["drop_col"] = pd.Series(["drop1", "drop2", "drop3"])
    feat_selector = feat.select_features(["adddate"]).set_output(transform="pandas")
    feature_df = feat_selector.fit_transform(test_df)
    assert feature_df.shape[1] == 1
    assert feature_df.shape[0] == 3
    assert list(feature_df.columns) == ["adddate"]


def test_get_dataset_indices():
    data = {"testcol": [1, 2, 3]}
    train_df = pd.DataFrame(data)
    train_df.index = [5, 6, 7]

    val_df = pd.DataFrame(data)
    val_df.index = [10, 11, 12]

    test_df = pd.DataFrame(data)
    test_df.index = [102, 101, 100]

    test_dict = feat.get_dataset_indices(train_df, val_df, test_df)
    check_dict = {
        "train": [5, 6, 7],
        "validation": [10, 11, 12],
        "test": [102, 101, 100],
    }
    assert test_dict == check_dict
