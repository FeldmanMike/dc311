"""
Download datasets and transform them to CSVs
"""

import argparse
import logging
import os

from dotenv import load_dotenv
import pandas as pd

from config.logging_config import setup_logging
import dc311.data.preprocess as prep


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

    logger.info("Fetching directory with data to be preprocessed...")
    if args.directory:
        raw_file_dir = args.directory
    else:
        project_dir = os.path.dirname(os.path.dirname(__file__))
        raw_file_dir = os.path.join(project_dir, "data", "raw")
    logger.info(f"Data to preprocess is saved in directory: {raw_file_dir}")

    logger.info("Fetching names of files to be preprocessed...")
    if args.files:
        raw_file_list = args.files
    else:
        raw_file_list = [f for f in os.listdir(raw_file_dir) if f.endswith(".csv")]
    logger.info(f"Files to be preprocessed are: {raw_file_list}")

    out_file_dir = os.path.join(project_dir, "data", "interim")
    logger.info(f"Preprocessed data to be output to: {out_file_dir}")
    for filename in raw_file_list:
        raw_file = os.path.join(raw_file_dir, filename)
        logger.info(f"Preprocessing {raw_file}")
        df = pd.read_csv(raw_file)

        # Perform preprocessing

        # Output to out_file_dir


if __name__ == "__main__":
    main()
