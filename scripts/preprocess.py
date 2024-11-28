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
        "-d",
        "--directory",
        type=str,
        required=False,
        help="Directory with data to be processed. If not provided, default directory"
        "is `dc311/data/raw/`.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        help="Names of files to be transformed. If none are provided, default is to "
        "transform all files in directory provided.",
    )
    args = parser.parse_args()

    if args.directory:
        raw_file_dir = args.directory
    else:
        project_dir = os.path.dirname(os.path.dirname(__file__))
        raw_file_dir = os.path.join(project_dir, "data", "raw")

    if args.files:
        raw_file_list = args.files
    else:




if __name__ == "__main__":
    main()
