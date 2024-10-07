"""
Test dc311/data/extract.py
"""

import os
import pytest

import pandas as pd

import dc311.data.transform as transform


def test_json_to_csv_conversion(test_json):
    try:
        curr_dir = os.path.dirname(__file__)
        json_path = os.path.join(curr_dir, "test.json")
        csv_path = os.path.join(curr_dir, "test.csv")
        transform.transform_json_to_csv(json_path, csv_path)

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
