"""
Set up fixtures for testing
"""

import json
import os
import pytest

import pandas as pd
import yaml


@pytest.fixture(scope="session")
def config():
    """
    Load config file from current directory
    """
    config_path = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


@pytest.fixture()
def test_json(scope="session"):
    """
    Create JSON file to test conversion to CSV
    """
    test_list = [
        {
            "attributes": {
                "OBJECTID": 1,
                "SERVICECALLCOUNT": 10,
                "SERVICEORDERSTATUS": "Open",
            },
            "geometry": {
                "x": 37.3,
                "y": 58.5,
            },
        },
        {
            "attributes": {
                "OBJECTID": 2,
                "SERVICECALLCOUNT": 0,
                "SERVICEORDERSTATUS": "Closed",
            },
            "geometry": {
                "x": 101.546,
                "y": 1.897,
            },
        },
        {
            "attributes": {
                "OBJECTID": 3,
                "SERVICECALLCOUNT": 35,
                "SERVICEORDERSTATUS": "Closed",
            },
            "geometry": {
                "x": 46.768,
                "y": 104.3,
            },
        },
    ]
    json_path = os.path.join(os.path.dirname(__file__), "test.json")
    with open(json_path, "w") as file:
        json.dump(test_list, file)


@pytest.fixture
def basic_dataframe():
    data_dict = {"COLUMN1": [1, 2, 3], "cOlUmN2": [4, 5, 6], "column3": [7, 8, 9]}
    return pd.DataFrame(data_dict)


@pytest.fixture
def datetime_conversion_df():
    data_dict = {
        "date_col_1": [1609479125000, 1609481125000],
        "date_col_2": [1609512550000, 1609509375000],
        "col_3": [1, 2],
    }
    return pd.DataFrame(data_dict)
