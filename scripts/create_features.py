"""
Engineer features and target
"""

import argparse
import logging
import os

from dotenv import load_dotenv
import pandas as pd
import yaml

from config.logging_config import setup_logging
import dc311.features as feat


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

    feature_pipe = feat.create_feature_engineering_pipeline(config["features"])
    feature_df = feature_pipe.fit_transform(dc311_df)

    # TODO - change preprocessing of days_to_resolve so that it is an integer


#    target_pipe = feat.create_target(
#                    df=dc311_df
#                    target_column=
#                    task=
#                    clf_threshold=
#                  )
#

if __name__ == "__main__":
    main()
