"""
Transform raw 311 data for exploratory data analysis and feature engineering
"""

import json
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def transform_json_to_csv(json_path: str, out_csv_path: str) -> None:
    """
    Transform DC 311 JSON file to a CSV

    Args:
        json_path: Path to JSON file to transform
        out_csv_path: Path to CSV file output by transformation

    Returns:
        None. CSV file is output to path provided.
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


def transform_column_names_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform column names of pandas dataframe to lowercase

    Args:
        df: A pandas DataFrame

    Returns:
        The pandas dataframe, with lowercase column names
    """
    df.columns = [col.lower() for col in df.columns]
    return df


def convert_columns_to_datetime(
    df: pd.DataFrame, time_column_list: List[str]
) -> pd.DataFrame:
    """
    Convert list of columns to datatime type

    Args:
        df: A pandas DataFrame
        time_column_list: List of column to convert to datetime type

    Returns:
        The pandas dataframe, with specified columns converted to datetime type
    """
    for col in time_column_list:
        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def create_days_to_resolve_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `days_to_resolve` field to DataFrame

    Args:
        df: A pandas DataFrame

    Returns:
        The pandas DataFrame, with new `days_to_resolve` column
    """
    df["days_to_resolve"] = df["resolutiondate"] - df["adddate"]
    return df


def process_ward_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process ward dataframe to clean up inconsistent data entry

    Args:
        df: A pandas DataFrame

    Returns:
        The pandas DataFrame with `ward` field processed
    """
    df["ward"] = df["ward"].replace(
        {
            "Ward 1": 1,
            "Ward 2": 2,
            "Ward 3": 3,
            "Ward 4": 4,
            "Ward 5": 5,
            "Ward 6": 6,
            "Ward 7": 7,
            "Ward 8": 8,
        }
    )
    return df
