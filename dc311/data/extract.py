'''
Extract 311 data from DC Data Portal
'''
import json
import logging
import os
import requests
from typing import Dict

logger = logging.getLogger(__name__)

def download_dataset_as_json(url: str, param_dict: Dict, outfile: str) -> None:
    '''
    Download a dataset from a given URL to a JSON.

    url: Full URL of the website from which to download data
    param_dict: Dictonary of parameters to append to the URL. View API
        documentation associated with URL for more detail on expected
        parameters
    outfile: Full path of JSON to be saved

    Returns:
        None. Outputs JSON file to the path provided.
    '''
    try:
        file_extension = os.path.splitext(outfile)[1] 
        if file_extension != ".json":
            logger.error(f"Invalid file extension for outfile: {file_extension}. "
                          "Expected a JSON extension.")
        
        logger.info(f"Retriving dataset from {url}...")
        logger.debug(f"Parameters passed to URL are: {param_dict}")
        response = requests.get(url, params=param_dict)
        logger.info(f"Dataset retrieved.")
        
        logger.info("Parsing the JSON response...")
        data = response.json()
        logger.info("JSON parsed.")
        
        logger.info(f"Dumping data to {outfile}...")
        with open(outfile, 'w') as file:
            json.dump(data, file, indent=4)
        logger.info("File saved.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)




