"""
Test dc311/data/extract.py
"""

import os

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import dc311.data.preprocess as prep


def test_json_to_csv_conversion(test_json):
    try:
        curr_dir = os.path.dirname(__file__)
        json_path = os.path.join(curr_dir, "test.json")
        csv_path = os.path.join(curr_dir, "test.csv")
        prep.transform_json_to_csv(json_path, csv_path)

        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 3

        col_list = ["OBJECTID", "SERVICECALLCOUNT", "SERVICEORDERSTATUS"]
        assert all(val in df.columns for val in col_list)

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(json_path):
            os.remove(json_path)


def test_transform_column_names_to_lowercase(basic_dataframe):
    df = prep.transform_column_names_to_lowercase(basic_dataframe)
    assert set(df.columns) == set(["column1", "column2", "column3"])


def test_convert_columns_to_datetime(datetime_conversion_df):
    df = prep.convert_columns_to_datetime(
        datetime_conversion_df, ["date_col_1", "date_col_2"]
    )
    assert is_datetime(df["date_col_1"])
    assert is_datetime(df["date_col_2"])
    assert not is_datetime(df["col_3"])
