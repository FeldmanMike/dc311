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
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
import xgboost as xgb


logger = logging.getLogger(__name__)


def split_data(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_split_dict: Dict,
    holdout_set_type: Optional[str] = "validation",
) -> Tuple:
    """
    Split data into train and test sets.

    Args:
        feature_df: DataFrame with features
        target_df: DataFrame with targets
        data_split_dict: Dictionary with indices on which to split data into train
            and test sets
        holdout_set_type: {"validation", "test"}
            Whether the holdout set should be from the validation or test set. If
            from the test set, then the training set is composed of the training
            and validation sets.

    Returns:
        Tuple with four elements: train features, train targets, test features,
        test targets
    """
    if holdout_set_type == "validation":
        X_train = feature_df.loc[data_split_dict["train"]]
        y_train = target_df.loc[data_split_dict["train"]]["target"]
        X_test = feature_df.loc[data_split_dict["validation"]]
        y_test = target_df.loc[data_split_dict["validation"]]["target"]
    elif holdout_set_type == "test":
        train_plus_val_idx = data_split_dict["train"] + data_split_dict["validation"]
        X_train = feature_df.loc[train_plus_val_idx]
        y_train = target_df.loc[train_plus_val_idx]["target"]
        X_test = feature_df.loc[data_split_dict["test"]]
        y_test = target_df.loc[data_split_dict["test"]]["target"]

    return X_train, y_train, X_test, y_test


def train_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Dict,
    task_type: str,
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
        task_type: {"regression", "classification"}
            Type of machine learning task
        model_type: {"logistic", "xgboost"}
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

    if task_type == "classification":
        if model_type == "logistic":
            steps.append(
                (
                    "classifier",
                    LogisticRegression(C=params["logreg_c"], random_state=random_seed),
                )
            )
        elif model_type == "xgboost":
            steps.append(
                (
                    "classifier",
                    xgb.XGBClassifier(
                        n_estimators=params["xgb_n_estimators"],
                        max_depth=params["xgb_max_depth"],
                        learning_rate=params["xgb_learning_rate"],
                        random_state=random_seed,
                    ),
                )
            )
        else:
            raise ValueError(
                f"model_type of {model_type} is not supported with classification task_type."
            )
    elif task_type == "regression":
        if model_type == "elasticnet":
            steps.append(
                "regressor",
                ElasticNet(
                    alpha=params["en_alpha"],
                    l1_ratio=params["en_l1_ratio"],
                    random_state=random_seed,
                ),
            )
        elif model_type == "xgboost":
            steps.append(
                (
                    "regressor",
                    xgb.XGBRegressor(
                        n_estimators=params["xgb_n_estimators"],
                        max_depth=params["xgb_max_depth"],
                        learning_rate=params["xgb_learning_rate"],
                        random_state=random_seed,
                    ),
                )
            )
        else:
            raise ValueError(
                f"model_type of {model_type} is not supported with regression task_type."
            )

    model_pipeline = Pipeline(steps)
    model_pipeline.fit(X, y)
    return model_pipeline


def objective(
    trial: optuna.trial.Trial,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_split_dict: Dict,
    task_type: str,
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
        task_type: {"regression", "classification"}
            Type of machine learning task
        model_type: {"logistic", "xgboost"}
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

        # xgboost can be used for regression or classification
        if model_type == "xgboost":
            max_depth_range = ranges["xgb_max_depth"]
            n_estimators_range = ranges["xgb_n_estimators"]
            lr_range = ranges["xgb_learning_rate"]
            params["xgb_max_depth"] = trial.suggest_int(
                "xgb_max_depth",
                int(max_depth_range["min"]),
                int(max_depth_range["max"]),
            )
            params["xgb_n_estimators"] = trial.suggest_int(
                "xgb_n_estimators",
                int(n_estimators_range["min"]),
                int(n_estimators_range["max"]),
            )
            params["xgb_learning_rate"] = trial.suggest_float(
                "xgb_learning_rate",
                float(lr_range["min"]),
                float(lr_range["max"]),
                log=True,
            )

        elif task_type == "classification":
            if model_type == "logistic":
                c_range = ranges["logreg_c"]
                params["logreg_c"] = trial.suggest_float(
                    "logreg_c", float(c_range["min"]), float(c_range["max"]), log=True
                )

            else:
                raise ValueError(
                    f"model_type {model_type} is not supported when task_type is classification."
                )
        elif task_type == "regression":
            if model_type == "elasticnet":
                alpha_range = ranges["en_alpha"]
                l1_ratio_range = ranges["en_l1_ratio"]
                params["en_alpha"] = trial.suggest_float(
                    "en_alpha",
                    float(alpha_range["min"]),
                    float(alpha_range["max"]),
                )
                params["en_l1_ratio"] = trial.suggest_float(
                    "en_l1_ratio",
                    float(l1_ratio_range["min"]),
                    float(l1_ratio_range["max"]),
                )
            else:
                raise ValueError(
                    f"model_type {model_type} is not supported when task_type is regression."
                )
        else:
            raise ValueError(
                f"task_type is {task_type} but task_type must be in"
                f"('classification', 'regression')"
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
