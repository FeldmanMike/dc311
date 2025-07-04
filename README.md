![DC Flag](assets/dc_311.png)

# DC 311 Requests: How Long Should They Take?
This project estimates resolution times for Washington DC 311 service requests using machine learning on historical data.

Try the live application on Streamlit: https://dc311eta.streamlit.app/

## Getting Started
Follow the steps below to launch the application locally:
1. Clone the repository
```
git clone https://github.com/FeldmanMike/dc311.git
cd dc311
```
2. Create and activate the environment
```
conda env create -f environment_dev.yml
conda activate dcenv
```
3. Launch the Streamlit app
```
streamlit run streamlit_app/app.py
```
4. Open the app
```
Your browser will open automatically, or visit:
http://localhost:8501
```

## Project Structure
```
dc311/              # Source code
models/             # Trained models and feature pipelines
streamlit_app/      # Streamlit app UI
scripts/            # Data extraction, processing, and model training scripts
data/               # Input and processed datasets (saved locally)
config/             # Config files informing ETL and model training
```

## Acknowledgments
All data was retrieved from the [DC Open Data Portal](https://opendata.dc.gov/). DC 311 image above is from the [DC 311 homepage](https://311.dc.gov/citizen/s/).

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this project with proper attribution.
