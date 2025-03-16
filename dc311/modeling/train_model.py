"""
Functionality to train model using optuna
"""

from datetime import datetime
import logging
from typing import Dict, Optional, Tuple

import mlflow
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline


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


def train_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Dict,
    model_type: Optional[str] = "logistic",
    pca: Optional[bool] = False,
    random_seed: Optional[int] = 0,
):
    """
    Fit an sklearn model pipeline object.

    Args:
        X: DataFrame of features
        y: DataFrame with targets corresponding to samples in X
        params: Dictionary of model hyperparameters
        model_type: {"logistic"}
            Type of model to be trained
        pca: Whether to apply principal component analysis to the feature data before
            passing the data to the classification model object
        random_seed: Seed provided to ensure reproducibility

    Returns:
        Fit sklearn model pipeline object
    """
    steps = []
    if pca:
        steps.append(
            (
                "pca",
                PCA(n_components=params["pca_n_components"], random_state=random_seed),
            )
        )

    if model_type == "logistic":
        steps.append(
            (
                "classifier",
                LogisticRegression(C=params["logreg_c"], random_state=random_seed),
            )
        )

    model_pipeline = Pipeline(steps)
    model_pipeline.fit(X, y)
    return model_pipeline


def objective(
    trial: optuna.trial.Trial,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_split_dict: Dict,
    model_type: Optional[str] = "logistic",
    pca: Optional[bool] = False,
    ranges: Optional[Dict] = None,
    random_seed: Optional[int] = 0,
) -> float:
    """
    Define objective function to optimize model

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
        ranges: Dictionary including ranges for hyperparameters for optuna to sample
            from
        random_seed: Seed provided to ensure reproducibility

    Returns:
        Brier score loss associated with training run
    """
    child_run_name = (
        f"child_run_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    with mlflow.start_run(nested=True, run_name=child_run_name):
        X_train, y_train, X_test, y_test = split_data(
            feature_df, target_df, data_split_dict
        )
        params = {}

        logreg_range = ranges["logreg_c"]
        params["logreg_c"] = trial.suggest_float(
            "logreg_c", float(logreg_range["min"]), float(logreg_range["max"]), log=True
        )

        if pca:
            params["pca_n_components"] = trial.suggest_int(
                "pca_n_components",
                float(ranges["pca_n_components"]["min"]),
                len(X_train.columns),
            )

        model = train_model(
            X=X_train,
            y=y_train,
            params=params,
            model_type=model_type,
            pca=pca,
            random_seed=random_seed,
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        brier_score = brier_score_loss(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)

        params["objective"] = "clf:min_brier_score"
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "brier_score_loss": brier_score,
                "roc_auc_score": roc_auc,
                "average_precision_score": average_precision,
            }
        )
        return brier_score
