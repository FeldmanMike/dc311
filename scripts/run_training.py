"""
Train, evaluate, and save model
"""

import argparse
from datetime import datetime
import joblib
import json
import logging
import os

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pandas as pd
import optuna
import yaml

from config.logging_config import setup_logging
from dc311.modeling import train_model as train


def main():
    setup_logging("model_training.log")
    logger = logging.getLogger(__name__)

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        help="Path to processed features and target. If not provided, default path"
        "is `dc311/data/processed/`",
    )
    parser.add_argument(
        "-s",
        "--save-model",
        action="store_true",
        help="Whether to save model file to save in `models/` folder. "
        "If not provided, model is not saved to `models/` folder",
    )
    parser.add_argument(
        "-r",
        "--retrain-with-test-set",
        action="store_true",
        required=False,
        default=False,
        help="Whether to retrain the model with the training, validation, and test "
        "sets. If not provided, then model is retrained with only the training "
        "and validation sets.",
    )
    args = parser.parse_args()

    try:
        config_path = os.getenv("DC_311_CONFIG_PATH")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        logger.debug("Fetching name of directory with processed data...")
        if args.input:
            data_dir = args.input
        else:
            project_dir = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(project_dir, "data", "processed")
        logger.debug(f"Processed data is saved in directory: {data_dir}")
        if config["task_type"] == "classification":
            feat_file_name = "processed_features_clf.csv"
            targ_file_name = (
                f"processed_target_clf_{str(config['target_threshold'])}.csv"
            )
            index_file_name = "dataset_indices_clf.json"
        elif config["task_type"] == "regression":
            feat_file_name = "processed_features_reg.csv"
            targ_file_name = "processed_target_reg.csv"
            index_file_name = "dataset_indices_reg.json"

        logger.info("Loading features, targets, and data split indices...")
        feature_df = pd.read_csv(
            os.path.join(data_dir, feat_file_name), index_col="objectid"
        )
        target_df = pd.read_csv(
            os.path.join(data_dir, targ_file_name), index_col="objectid"
        )
        with open(os.path.join(data_dir, index_file_name), "r") as f:
            data_split_dict = json.load(f)
        logger.info("Data loaded.")

        logger.info("Starting trials...")
        optuna.logging.enable_propagation()

        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(config["experiment_name"])
        parent_run_name = f"parent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=parent_run_name) as run:
            sampler = optuna.samplers.TPESampler(seed=config["random_seed"])
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda trial: train.objective(
                    trial=trial,
                    feature_df=feature_df,
                    target_df=target_df,
                    data_split_dict=data_split_dict,
                    task_type=config["task_type"],
                    model_type=config["model_type"],
                    pca=config["pca"],
                    ranges=config["ranges"],
                    random_seed=config["random_seed"],
                ),
                n_trials=config["n_trials"],
            )
            logger.info("Trials complete!")
            logger.info("Getting best model...")
            best_params = study.best_trial.params
            logger.info(f"Best params are: {best_params}")

            logger.info("Evaluating best model on test set...")
            X_train, y_train, X_test, y_test = train.split_data(
                feature_df=feature_df,
                target_df=target_df,
                data_split_dict=data_split_dict,
                holdout_set_type="test",
            )
            best_model = train.train_model(
                X=X_train,
                y=y_train,
                params=best_params,
                task_type=config["task_type"],
                model_type=config["model_type"],
                pca=config["pca"],
                random_seed=config["random_seed"],
            )

            logger.info("Logging best model...")
            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_brier_score", study.best_trial.value)
            metric_dict = train.evaluate_model(
                best_model, X_test, y_test, config["task_type"]
            )

            if config["task_type"] == "classification":
                mlflow.log_metrics(
                    {
                        "test_brier_score_loss": metric_dict["brier_score_loss"],
                        "test_roc_auc_score": metric_dict["roc_auc_score"],
                        "test_average_precision_score": metric_dict[
                            "average_precision_score"
                        ],
                    }
                )
                logger.info(
                    f"Test metrics of best model:\n"
                    f"Brier score: {metric_dict['brier_score_loss']:.4f}, "
                    f"ROC AUC: {metric_dict['roc_auc_score']:.4f}, "
                    f"Average precision: {metric_dict['average_precision_score']:.4f}"
                )

            elif config["task_type"] == "regression":
                mlflow.log_metrics(
                    {
                        "test_mean_squared_error": metric_dict["mean_squared_error"],
                        "test_mean_absolute_error": metric_dict["mean_absolute_error"],
                        "test_median_absolute_error": metric_dict[
                            "median_absolute_error"
                        ],
                        "test_r2_score": metric_dict["r2_score"],
                    }
                )
                logger.info(
                    f"Test metrics of best model:\n"
                    f"Mean squared error: {metric_dict['mean_squared_error']:.4f}, "
                    f"Mean absolute error: {metric_dict['mean_absolute_error']:.4f}, "
                    f"Median absolute error: {metric_dict['median_absolute_error']:.4f}, "
                    f"R2: {metric_dict['r2_score']:.4f}"
                )
            else:
                raise ValueError(
                    f"task_type is {config['task_type']}, but task_type must be "
                    f"in ('classification', 'regression')."
                )

            if args.retrain_with_test_set:
                logger.info(
                    "Retraining model using training, validation, and test sets..."
                )
                best_model = train.train_model(
                    X=feature_df,
                    y=target_df,
                    params=best_params,
                    task_type=config["task_type"],
                    model_type=config["model_type"],
                    pca=config["pca"],
                    random_seed=config["random_seed"],
                )
                metric_dict = train.evaluate_model(
                    best_model, X_test, y_test, config["task_type"]
                )
                if config["task_type"] == "classification":
                    mlflow.log_metrics(
                        {
                            "train_brier_score_loss": metric_dict["brier_score_loss"],
                            "train_roc_auc_score": metric_dict["roc_auc_score"],
                            "train_average_precision_score": metric_dict[
                                "average_precision_score"
                            ],
                        }
                    )
                    logger.info(
                        f"Training metrics of retrained best model:\n"
                        f"Brier score: {metric_dict['brier_score_loss']:.4f}, "
                        f"ROC AUC: {metric_dict['roc_auc_score']:.4f}, "
                        f"Average precision: {metric_dict['average_precision_score']:.4f}"
                    )

                elif config["task_type"] == "regression":
                    mlflow.log_metrics(
                        {
                            "train_mean_squared_error": metric_dict[
                                "mean_squared_error"
                            ],
                            "train_mean_absolute_error": metric_dict[
                                "mean_absolute_error"
                            ],
                            "train_median_absolute_error": metric_dict[
                                "median_absolute_error"
                            ],
                            "train_r2_score": metric_dict["r2_score"],
                        }
                    )
                    logger.info(
                        f"Training metrics of retrained best model:\n"
                        f"Mean squared error: {metric_dict['mean_squared_error']:.4f}, "
                        f"Mean absolute error: {metric_dict['mean_absolute_error']:.4f}, "
                        f"Median absolute error: {metric_dict['median_absolute_error']:.4f}, "
                        f"R2: {metric_dict['r2_score']:.4f}"
                    )
                else:
                    raise ValueError(
                        f"task_type is {config['task_type']}, but task_type must be "
                        f"in ('classification', 'regression')."
                    )

            logger.info("Model logging complete!")
            if args.save_model:
                logger.info("Saving model...")
                project_dir = os.path.dirname(os.path.dirname(__file__))
                if config["task_type"] == "classification":
                    model_name = (
                        f"under_{str(config['target_threshold'])}_day_model.joblib"
                    )
                elif config["task_type"] == "regression":
                    model_name = "num_days_model.joblib"
                model_path = os.path.join(
                    project_dir, "models", args.output, model_name
                )
                joblib.dump(best_model, model_path)
                logger.info(f"Model saved at: {model_path}")
            else:
                link_to_model = f"{config['tracking_uri']}/{run.info.experiment_id}/{run.info.run_id}/artifacts/best_model"
                logger.info("Load model with following call:")
                logger.info(f"mlflow.sklearn.load_model('{link_to_model}')")
    except Exception as e:
        logger.exception(f"There was an error: {e}")
        raise


if __name__ == "__main__":
    main()
