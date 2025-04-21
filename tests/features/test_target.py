"""
Test dc311/features/target.py
"""

import pytest

import pandas as pd

import dc311.features.target as targ


def test_create_target_assertion_error():
    """Make sure create_target() raises AssertionError with bad input"""
    with pytest.raises(AssertionError):
        targ.create_target(
            df=pd.DataFrame(), target_column="days", task="bad input", clf_threshold=0
        )


def test_create_target_regression(test_dataframe):
    """Test create_target() with regression task"""
    df = targ.create_target(df=test_dataframe, target_column="days", task="regression")
    assert df["target"].to_list() == [2, 8, 3, 7]


def test_create_target_classification(test_dataframe):
    """Test create_target() with classification task"""
    df = test_dataframe.set_index("col1")
    df = targ.create_target(
        df=test_dataframe,
        target_column="days",
        task="classification",
        clf_threshold=3,
    )
    expected = [0, 1, 1, 1, 0, 1, 1, 1]
    assert df["target"].to_list() == [float(i) for i in expected]
