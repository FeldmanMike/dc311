"""
Engineer features and target
"""

import argparse
from datetime import datetime
import json
import logging
import os

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
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
        "-o",
        "--output",
        type=str,
        required=False,
        help="Name of model file to save in `models/` folder. If not provided, "
             "default name of file is model_<timestamp>.joblib",
    )
    parser.add_argument(
        "-r",
        "--retrain-with-test",
        action="store_true",
        required=False,
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

        logger.info("Loading features, targets, and data split indices...")
        feature_df = pd.read_csv(
            os.path.join(data_dir, "processed_features.csv"), index_col="objectid"
        )
        target_df = pd.read_csv(
            os.path.join(data_dir, "processed_target.csv"), index_col="objectid"
        )
        with open(os.path.join(data_dir, "dataset_indices.json"), "r") as f:
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
                holdout_set_type="test"
            )
            best_model = train.train_model(
                X=X_train,
                y=y_train,
                params=best_params,
                model_type=config["model_type"],
                pca=config["pca"],
                random_seed=config["random_seed"],
            )
            y_proba = best_model.predict_proba(X_test)[:, 1]
            brier_score = brier_score_loss(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            average_precision = average_precision_score(y_test, y_proba)

            logger.info("Logging best model...")
            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_brier_score", study.best_trial.value)
            mlflow.log_metrics(
                {
                    "test_brier_score_loss": brier_score,
                    "test_roc_auc_score": roc_auc,
                    "test_average_precision_score": average_precision,
                }
            )
            
            # TODO - save out model to file and remove link to model
            # if no output file provided, save out with timestamp
            if args.retrain_with_test:
                holdout_set_type = "test"
            else:
                holdout_set_type = "validation"

            logger.info("Model logging complete!")
            logger.info("Load best model with following call:")
            link_to_model = f"{config['tracking_uri']}/{run.info.experiment_id}/{run.info.run_id}/artifacts/best_model"
            logger.info(f"mlflow.sklearn.load_model('{link_to_model}')")
    except Exception as e:
        logger.exception(f"There was an error: {e}")
        raise


if __name__ == "__main__":
    main()
