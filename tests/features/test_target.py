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
