"""
Set up fixtures for testing
"""

import pytest

import numpy as np
import pandas as pd


@pytest.fixture
def test_dataframe():
    data = {
        "col1": [1, 2, 3, 4, 5, 6, 7, 8],
        "days": [2, 8, pd.NaT, np.nan, 3, None, 7, ""],
    }
    return pd.DataFrame(data)


@pytest.fixture
def time_dataframe():
    data = {
        "year": [2021, 2022, 2023],
        "month": [1, 4, 7],
        "day": [10, 15, 20],
        "hour": [5, 14, 21],
        "minute": [30, 15, 45],
        "second": [0, 15, 30],
    }
    df = pd.DataFrame(data)
    df["adddate"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
            "minute": df["minute"],
            "second": df["second"],
        }
    )
    return df
