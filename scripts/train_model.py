"""
Engineer features and target
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv
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
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: train.objective(
                trial=trial,
                tracking_uri=config["tracking_uri"],
                experiment_name=config["experiment_name"],
                feature_df=feature_df,
                target_df=target_df,
                data_split_dict=data_split_dict,
                pca=config["pca"],
            ),
            n_trials=config["n_trials"],
        )
        logger.info("Trials complete!")
    except Exception as e:
        logger.exception(f"There was an error: {e}")
        raise


if __name__ == "__main__":
    main()
