"""
Extract 311 data from DC Data Portal
"""

import json
import logging
import os
import requests
from typing import Dict

logger = logging.getLogger(__name__)


def download_dataset_as_json(url: str, param_dict: Dict, outfile: str,
                             max_records: int = None) -> None:
    """
    Download a dataset from a given URL to a JSON.

    url: Full URL of the website from which to download data
    param_dict: Dictonary of parameters to append to the URL. View API
        documentation associated with URL for more detail on expected
        parameters
    max_records: Maximum number of records to extract. If None or negative, then
        the maximum number of records possible will be extracted
    outfile: Full path of JSON to be saved

    Returns:
        None. Outputs JSON file to the path provided.
    """
    try:
        file_extension = os.path.splitext(outfile)[1]
        if file_extension != ".json":
            logger.error(
                f"Invalid file extension for outfile: {file_extension}. "
                "Expected a JSON extension."
            )

        logger.info(f"Retriving dataset from {url}...")
        logger.debug(f"Parameters passed to URL are: {param_dict}")

        all_records = []

        # We can only retrieve 1000 records at a time, so we will need many API calls
        # to retrieve all the data
        while True:
            response = requests.get(url, params=param_dict)
            data = response.json()

            # Check if records are present
            if "features" not in data or len(data["features"])== 0:
                break

            record_list = data["features"]
            all_records.extend(record_list)

            if len(all_records) % 50000 == 0:
                logging.info(f"Retrieved {len(all_records)} records...")

            # If fewer records are returned than resultRecordCount, we have retrieved
            # all the data
            if len(record_list) < param_dict["resultRecordCount"]:
                break
            
            # Ensure we do not exceed max_records if it is provided
            if (max_records is not None) and (len(all_records) >= max_records):
                all_records = all_records[:max_records]
                break

            param_dict["resultOffset"] += param_dict["resultRecordCount"]

        logger.info(f"Retrieved all {len(all_records)} records.")
        logger.info(f"Dumping data to {outfile}...")
        with open(outfile, "w") as file:
            json.dump(data, file, indent=4)
        logger.info("File saved.")
    except requests.exceptions.JSONDecodeError as e:
        # TODO - add record number of invalid record
        logger.info("Skipping invalid record...")
        continue
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
