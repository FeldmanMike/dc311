"""
Engineer features and target
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv
import pandas as pd
import yaml

from config.logging_config import setup_logging
import dc311.features.features as feat
import dc311.features.target as targ


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        help="Path to preprocessed data file. If not provided, default path"
        "is `dc311/data/interim/dc_311_preprocessed_data.csv`.",
    )
    args = parser.parse_args()

    config_path = os.getenv("DC_311_CONFIG_PATH")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    logger.info("Fetching name of directory with data to be preprocessed...")
    if args.input:
        data_path = args.input
    else:
        project_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(
            project_dir, "data", "interim", "dc_311_preprocessed_data.csv"
        )
    logger.info(f"Data to preprocess is saved in directory: {data_path}")

    out_file_dir = os.path.join(project_dir, "data", "processed")
    dc311_df = pd.read_csv(data_path)
    dc311_df = dc311_df.set_index("objectid")
    dc311_df["adddate"] = pd.to_datetime(dc311_df["adddate"])
    logger.info(f"Columns in dataframe read from {data_path} are: {dc311_df.columns}")

    logger.info("Splitting data into training, validaton, and test sets...")
    train_df = dc311_df[dc311_df["adddate"].dt.year.isin(config["train_year"])]
    validation_df = dc311_df[
        dc311_df["adddate"].dt.year.isin(config["validation_year"])
    ]
    test_df = dc311_df[dc311_df["adddate"].dt.year.isin(config["test_year"])]
    logger.info("Data split complete.")

    feature_pipe = feat.create_feature_engineering_pipeline(config["features"])

    logger.info("Creating features for training set...")
    train_df = feature_pipe.fit_transform(train_df)

    logger.info("Creating features for validation set...")
    validation_df = feature_pipe.transform(validation_df)

    logger.info("Creating features for test set...")
    test_df = feature_pipe.transform(test_df)
    logger.info("Features created successfully.")

    logger.info("Creating target...")
    target_df = targ.create_target(
        df=dc311_df,
        target_column="days_to_resolve",
        task="classification",
        clf_threshold=4,
    )
    logger.info("Target created successfully.")

    logger.info("Getting dataset indices...")
    dataset_indices = feat.get_dataset_indices(train_df, validation_df, test_df)
    logger.info("Dataset indices retrieved.")

    logger.info("Concatenating train, validation, and test sets back together...")
    feature_df = pd.concat([train_df, validation_df, test_df])
    logger.info("Concatenation complete.")

    logger.info(f"Saving features, target, and indices to {out_file_dir}")
    feature_df.to_csv(os.path.join(out_file_dir, "processed_features.csv"))
    target_df.to_csv(os.path.join(out_file_dir, "processed_target.csv"))

    outfile = os.path.join(out_file_dir, "dataset_indices.json")
    with open(outfile, "w") as json_file:
        json.dump(dataset_indices, json_file, indent=4)
    logger.info("Save complete.")


if __name__ == "__main__":
    main()
