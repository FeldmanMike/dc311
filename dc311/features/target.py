"""
Create target for model
"""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import Binarizer


def create_target(
    df: pd.DataFrame, target_column: str, task: str, clf_threshold: Optional[int]
) -> pd.Series:
    """
    Create target for model.

    Args:
        df: pandas DataFrame with model target
        target_column: Name of column to process as model target
        task: Machine learning task to be completed. Either 'regression'
            or 'classification'
        clf_threshold: If task = 'classification', then choose the 0/1
        threshold at which the target will be binarized

    Returns:
        pandas Series that will be a target for the model
    """
    task_values = ({"regression", "classification"},)
    assert task in task_values, f"Value of task must be in {task_values}"

    if task == "regression":
        return df[target_column]
    binarizer = Binarizer(threshold=clf_threshold)
    return pd.Series(
        binarizer.fit_transform(df[[target_column]]).ravel(), name=target_column
    )
