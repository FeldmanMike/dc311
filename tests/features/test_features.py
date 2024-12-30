"""
Test dc311/features/features.py
"""

import dc311.features.features as feat


def test_create_year_feature(time_dataframe):
    return_df = feat.create_year_feature(time_dataframe)
    assert return_df["add_year"].to_list() == time_dataframe["year"].to_list()


def test_create_month_feature(time_dataframe):
    return_df = feat.create_month_feature(time_dataframe)
    assert return_df["add_month"].to_list() == time_dataframe["month"].to_list()


def test_create_quarter_feature(time_dataframe):
    return_df = feat.create_quarter_feature(time_dataframe)
    assert return_df["add_quarter"].to_list() == [1, 2, 3]


def test_create_day_feature(time_dataframe):
    return_df = feat.create_day_feature(time_dataframe)
    assert return_df["add_day"].to_list() == [6, 4, 3]
