"""
Preprocess data before engineering features
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

    try:
        logger.debug("Fetching name of directory with data to be preprocessed...")
        if args.directory:
            raw_file_dir = args.directory
        else:
            project_dir = os.path.dirname(os.path.dirname(__file__))
            raw_file_dir = os.path.join(project_dir, "data", "raw")
        logger.debug(f"Data to preprocess is saved in directory: {raw_file_dir}")

        logger.debug("Fetching names of files to be preprocessed...")
        if args.files:
            raw_file_list = args.files
        else:
            raw_file_list = [f for f in os.listdir(raw_file_dir) if f.endswith(".csv")]
        logger.debug(f"Files to be preprocessed are: {raw_file_list}")

        outfile_dir = os.path.join(project_dir, "data", "interim")
        final_outfile_path = os.path.join(outfile_dir, "dc_311_preprocessed_data.csv")
        if os.path.exists(final_outfile_path):
            logger.debug(f"{final_outfile_path} exists! Bypassing preprocessing.")
        else:
            df_list = []
            for filename in raw_file_list:
                raw_file = os.path.join(raw_file_dir, filename)
                logger.info(f"Reading {raw_file}...")
                df = pd.read_csv(raw_file)
                logger.info(f"{raw_file} read.")
                logger.info(f"Preprocessing {raw_file}")
                df = prep.transform_column_names_to_lowercase(df)

                time_columns = [
                    "adddate",
                    "resolutiondate",
                    "serviceduedate",
                    "serviceorderdate",
                    "inspectiondate",
                ]
                df = prep.convert_columns_to_datetime(df, time_columns)

                df = prep.create_days_to_resolve_field(df)
                df = prep.process_ward_field(df)
                outfile_path = os.path.join(outfile_dir, filename)
                logger.info(
                    f"Preprocessing of {raw_file} complete. Outputting data to {outfile_path}..."
                )
                df.to_csv(outfile_path, index=False)
                logger.info(f"File successfully output to {outfile_path}.")
                df_list.append(df)
            logging.info("Stacking dataframes...")
            df_all_years = pd.concat(df_list)
            logging.info("Dataframes successfully stacked.")
            logger.info(f"Saving stacked dataframe as CSV to {final_outfile_path}.")
            df_all_years.to_csv(final_outfile_path, index=False)
            logger.info("CSV successfully saved.")
    except Exception as e:
        logger.exception(f"There was an error: {e}")
        raise


if __name__ == "__main__":
    main()
