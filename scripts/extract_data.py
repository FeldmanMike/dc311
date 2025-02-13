"""
Download datasets and transform them to CSVs
"""

import argparse
import copy
import logging
import os
import yaml

from dotenv import load_dotenv

from config.logging_config import setup_logging
from dc311.data.extract import download_dataset_as_json
from dc311.data.preprocess import transform_json_to_csv


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=int,
        help="Years for which data should be pulled and converted. Consult the "
        "config file for years available.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force download all years provided, if `-y` flag "
        "provided, or all years of data that can possibly "
        "be downloaded, if `-y` flag not provided, "
        "regardless of whether data has already been downloaded.",
    )
    args = parser.parse_args()

    try:
        config_path = os.getenv("DC_311_CONFIG_PATH")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        project_dir = os.path.dirname(os.path.dirname(__file__))
        raw_file_dir = os.path.join(project_dir, "data", "raw")
        endpoints = config["dc_311_data_api_endpoints"]
        query_params = config["api_query_parameters"]

        possible_years = [year for year in endpoints]
        if args.years:
            year_list = args.years
        else:
            year_list = possible_years

        logger.debug(f"Years to be cycled through: {year_list}")
        for year in year_list:
            json_filename = os.path.join(raw_file_dir, f"dc_311_{str(year)}_data.json")
            if args.force or not os.path.exists(json_filename):
                logger.info(f"Getting data from year: {year}")

                # Copy query_params to ensure we do not modify original params
                download_dataset_as_json(
                    endpoints[year], copy.deepcopy(query_params), json_filename
                )
            else:
                logger.info(
                    f"Dataset for {year} already downloaded to {json_filename}. Skipping "
                    "download..."
                )

            csv_filename = os.path.join(raw_file_dir, f"dc_311_{str(year)}_data.csv")
            if args.force or not os.path.exists(csv_filename):
                logger.info(f"Transforming {json_filename} to CSV at {csv_filename}...")
                transform_json_to_csv(json_filename, csv_filename)
            else:
                logger.info(
                    f"File {csv_filename} already exists. Skipping data transformation..."
                )
    except Exception as e:
        logger.exception(f"There was an error: {e}")
        raise


if __name__ == "__main__":
    main()
