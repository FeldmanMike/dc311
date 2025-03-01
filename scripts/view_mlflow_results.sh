#!/bin/bash

set -e

mlflow ui --backend-store-uri data/mlruns
echo "View results in browser at http://localhost:5000"
