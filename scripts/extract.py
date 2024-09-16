"""
Download datasets
"""
import argparse
import logging
import os
import yaml

from dotenv import load_dotenv

from config.logging_config import setup_logging
from dc311.data.extract import download_dataset_as_json


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
        help="Years for which data should be pulled. Consult the "
        "config file for years available.",
    )
    parser.add_argument(
        "-f",
        "--force",
        type="store_true",
        help="Force download all years provided, if `-y` flag "
        "provided, or all years of data that can possibly "
        "be downloaded, if `-y` flag not provided, "
        "regardless of whether data has already been downloaded.",
    )
    args = parser.parse_args()

    config_path = os.getenv("DC_311_CONFIG_PATH")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    project_dir = os.path.dirname(os.path.dirname(__file__))
    outfile_dir = os.path.join(project_dir, "data", "raw")
    endpoints = config["dc_311_data_api_endpoints"]
    query_params = config["api_query_parameters"]
    
    possible_years = [year for year in endpoints]
    if args.years:
        year_list = args.years
    else:
        year_list = possible_years
    
    logger.debug(f"Years to be cycled through: {year_list}")
    for year in year_list:
        filename = os.path.join(outfile_dir, f"dc_311_{str(year)}_data.json")
        if args.force or not os.path.exists(filename):
            logger.info(f"Getting data from year: {year}")
            download_dataset_as_json(endpoints[year], query_params, filename)


if __name__ == "__main__":
    main()
