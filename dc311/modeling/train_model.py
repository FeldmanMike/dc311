"""
Functionality to train model using optuna
"""

import logging
from typing import Dict, Tuple

import mlflow
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


logger = logging.getLogger(__name__)


def split_data(
    feature_df: pd.DataFrame, target_df: pd.DataFrame, data_split_dict: Dict
) -> Tuple:
    """
    Split data into train and test sets.

    Args:
        feature_df: DataFrame with features
        target_df: DataFrame with targets
        data_split_dict: Dictionary with indices on which to split data into train
            and test sets

    Returns:
        Tuple with four elements: train features, train targets, test features,
        test targets
    """
    X_train = feature_df.loc[data_split_dict["train"]]
    y_train = target_df.loc[data_split_dict["train"]]["target"]
    X_test = feature_df.loc[data_split_dict["validation"]]
    y_test = target_df.loc[data_split_dict["validation"]]["target"]
    return X_train, y_train, X_test, y_test


def objective_logreg(
    trial: optuna.trial.Trial,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_split_dict: Dict,
) -> float:
    """
    Define objective function to maximize logistic regression model
    """
    with mlflow.start_run(nested=True):
        X_train, y_train, X_test, y_test = split_data(
            feature_df, target_df, data_split_dict
        )
        params = {}
        params["logreg_c"] = trial.suggest_float("logrec_c", 1e-10, 1e10, log=True)
        params["objective"] = "clf:min_brier_score"

        clf = LogisticRegression(C=params["logreg_c"], random_state=0)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        brier_score = brier_score_loss(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metric(brier_score_loss)
        mlflow.log_metrics(
            {
                "brier_score_loss": brier_score,
                "roc_auc_score": roc_auc,
                "average_precision_score": average_precision,
            }
        )
        return brier_score_loss
