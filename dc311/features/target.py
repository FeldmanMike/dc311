"""
Create target for model
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer

logger = logging.getLogger(__name__)


def create_target(
    df: pd.DataFrame, target_column: str, task: str, clf_threshold: Optional[int] = None
) -> pd.DataFrame:
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
        threshold at which the target will be binarized. Values below or equal
        to this value will be set equal to zero; those higher will be set to 1.

    Returns:
        pandas DataFrame that includes only one column: a target for the model called "target"
    """
    task_values = set(["regression", "classification"])
    assert task in task_values, f"Value of task '{task}' not in {task_values}."

    if task == "regression":
        df["target"] = df[target_column].replace("", np.nan)
        return df[df["target"].notna()][["target"]]

    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    imputer = SimpleImputer(strategy="constant", fill_value=999999)
    df[["target"]] = imputer.fit_transform(df[[target_column]])

    binarizer = Binarizer(threshold=clf_threshold)
    df[["target"]] = binarizer.fit_transform(df[["target"]])

    return df[["target"]]
