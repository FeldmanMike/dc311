"""
Create target for model
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer


def create_target(
    df: pd.DataFrame, target_column: str, task: str, clf_threshold: Optional[int] = None
) -> pd.Series:
    """
    Create target for model. For regression tasks, missing values are dropped
    and for classification tasks, missing values are kept and imputed to 999999
    (under the assumption that missingness indicates incomplete 311 cases).

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
    task_values = set(["regression", "classification"])
    assert task in task_values, f"Value of task '{task}' not in {task_values}."

    if task == "regression":
        return df[target_column].replace("", np.nan).dropna()

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=999999)),
            ("binarizer", Binarizer(threshold=clf_threshold)),
        ]
    )
    return pd.Series(
        pipe.fit_transform(df[[target_column]]).ravel(), name=target_column
    )
