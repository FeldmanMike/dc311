'''
Download datasets
'''
import configparser
import os

from dotenv import load_dotenv

from config.logging_config import setup_logging
from dc311.data.extract import download_dataset_as_json


def main():

    setup_logging()
    
    load_dotenv()
    config_path = os.getenv('DC_311_CONFIG_PATH')

    config = configparser.ConfigParser()
    config.read(config_path)

    project_dir = os.path.dirname(os.path.dirname(__file__))
    outfile_dir = os.path.join(project_dir, "data", "raw")
    endpoints = config['dc_311_data_api_endpoints']
    query_params = config['api_query_parameters']

    # Download 2023 DC 311 dataset
    outfile_2023 = os.path.join(outfile_dir, "dc_311_2023_data.json")
    download_dataset_as_json(endpoints[2023], query_params, outfile_2023)

if __name__ == '__main__':
    main()

