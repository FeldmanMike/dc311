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
        "target": [2, 8, pd.NaT, np.nan, 3, None, 7, ""],
    }
    return pd.DataFrame(data)
