#!/bin/bash

set -e

python3 -m scripts.extract_data
python3 -m scripts.preprocess_data
python3 -m scripts.create_features
python3 -m scripts.train_model
