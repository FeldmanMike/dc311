"""
Transform raw 311 data for exploratory data analysis and feature engineering
"""
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def transform_json_to_csv(json_path: str, out_csv_path: str) -> None:
    """
    Transform DC 311 JSON file to a CSV
    """
    logger.info("Loading JSON from file...")
    with open(json_path, "r") as file:
        data_dict = json.load(file)
    logger.info("JSON loaded.")

    logger.info("Retrieving records...")
    all_records = []
    for record in data_dict:
        all_records.append(record["attributes"])
    logger.info("Records retrieved.")

    logger.info("Creating DataFrame from records...")
    df = pd.DataFrame.from_records(all_records)
    logger.info("Created DataFrame.")

    logger.info("Exporting DataFrame to CSV...")
    df.to_csv(out_csv_path, index=False)
    logger.info("Exported to CSV.")

    return None
