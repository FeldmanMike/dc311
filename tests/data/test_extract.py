"""
Test dc311/data/extract.py
"""

import os
import pytest

import dc311.data.extract as extract


@pytest.mark.parametrize("year", [2022])
def test_data_extraction(config, year):
    url = config["dc_311_data_api_endpoints"][year]
    outfile = os.path.join(os.path.dirname(__file__), f"test_{str(year)}_311_data.json")

    try:
        extract.download_dataset_as_json(
            url=url,
            param_dict=config["api_query_parameters"],
            outfile=outfile,
            max_records=config["max_num_records"],
        )
        assert os.path.exists(outfile)
    finally:
        if os.path.exists(outfile):
            os.remove(outfile)
