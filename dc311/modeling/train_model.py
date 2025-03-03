"""
Functionality to train model using optuna
"""

import logging
from typing import Dict, Optional, Tuple

import mlflow
import optuna
import pandas as pd
from sklearn.decomposition import PCA
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


def objective(
    trial: optuna.trial.Trial,
    tracking_uri: str,
    experiment_name: str,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_split_dict: Dict,
    model_type: Optional[bool] = "logistic",
    pca: Optional[bool] = False,
) -> float:
    """
    Define objective function to maximize logistic regression model

    Args:
        trial: optuna Trial object on which to optimize
        tracking_uri: mlflow tracking uri that defines where logged data will be stored
        experiment_name: Name of experiment associated with mlflow runs
        feature_df: DataFrame of features for model training run
        target_df: DataFrame with targets associated with features
        data_split_dict: Dictionary that specifies indices for samples associated with
            train and test sets
        model_type: {"logistic"}
            Type of model to train
        pca: Whether to apply principal component analysis to the feature data before
            passing the data to the classification model object

    Returns:
        Brier score loss associated with training run
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(nested=True):
        X_train, y_train, X_test, y_test = split_data(
            feature_df, target_df, data_split_dict
        )
        params = {}
        params["logreg_c"] = trial.suggest_float("logrec_c", 1e-10, 1e10, log=True)
        params["objective"] = "clf:min_brier_score"

        if pca:
            params["pca_n_components"] = trial.suggest_int(
                "pca_n_components", 1, len(X_train.columns)
            )
            pca = PCA(n_components=params["pca_n_components"], random_state=0)
            X_train = pca.fit_transform(X_train)

        clf = LogisticRegression(C=params["logreg_c"], random_state=0)
        clf.fit(X_train, y_train)

        # Get train set metrics
        y_proba = clf.predict_proba(X_train)[:, 1]
        train_brier_score = brier_score_loss(y_train, y_proba)
        train_roc_auc = roc_auc_score(y_train, y_proba)
        train_average_precision = average_precision_score(y_train, y_proba)

        if pca:
            X_test = pca.transform(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Get test set metrics
        brier_score = brier_score_loss(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "brier_score_loss": brier_score,
                "roc_auc_score": roc_auc,
                "average_precision_score": average_precision,
                "train_set_brier_score_loss": train_brier_score,
                "train_set_roc_auc_score": train_roc_auc,
                "train_set_average_precision_score": train_average_precision,
            }
        )
        return brier_score
