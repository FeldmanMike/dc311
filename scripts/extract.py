'''
Download datasets
'''
import os

from config.logging_config import setup_logging
from dc311.data.extract import download_dataset_as_json


def main():

    setup_logging()

    url = (
        "https://maps2.dcgis.dc.gov/dcgis/rest/services/"
        "DCGIS_DATA/ServiceRequests/MapServer/15/query"
    )
    
    params = {
        "where": "1=1",
        "outFields": "*",
        "resultRecordCount": 30,
        "f": "json"
    }

    project_dir = os.path.dirname(os.path.dirname(__file__))
    outfile = os.path.join(project_dir, "data", "raw", "test_file.json")
    download_dataset_as_json(url, params, outfile)

if __name__ == '__main__':
    main()

