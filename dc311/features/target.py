"""
Create target for model
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, FunctionTransformer

logger = logging.getLogger(__name__)


def to_numeric(X):
    return X.apply(pd.to_numeric, errors="coerce")


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
        return df[df["target"].notna()]

    pipe = Pipeline(
        [
            ("to_numeric", FunctionTransformer(to_numeric)),
            ("imputer", SimpleImputer(strategy="constant", fill_value=999999)),
            ("binarizer", Binarizer(threshold=clf_threshold)),
        ]
    )
    df["target"] = pd.Series(
        pipe.fit_transform(df[[target_column]]).ravel(), name=target_column
    )
    return df[["target"]]
