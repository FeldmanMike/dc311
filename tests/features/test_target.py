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
            df=pd.DataFrame(), target_column="target", task="bad input", clf_threshold=0
        )


def test_create_target_regression(test_dataframe):
    """Test create_target() with regression task"""
    test_dataframe["final_target"] = targ.create_target(
        df=test_dataframe, target_column="target", task="regression"
    )
    assert test_dataframe["final_target"].to_list() == [2, 8, 3, 7]
