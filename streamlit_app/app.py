"""
Develop Streamlit app
"""

from datetime import date
import json
import os
import os.path as osp
import sys

from dotenv import load_dotenv
import joblib
import pandas as pd
import streamlit as st
import yaml


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

load_dotenv()
config_path = os.getenv("DC_311_CONFIG_PATH")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

under_21_day_model = joblib.load("models/under_21_day_model.joblib")
under_5_day_model = joblib.load("models/under_5_day_model.joblib")
num_days_model = joblib.load("models/num_days_model.joblib")
feature_pipe_clf = joblib.load("models/feature_pipeline_clf.joblib")
feature_pipe_reg = joblib.load("models/feature_pipeline_reg.joblib")

with open("streamlit_app/request_categories.json") as f:
    REQUEST_TYPES = json.load(f)
WARDS = [f"Ward {i}" for i in range(1, 9)]

st.title("DC 311 Request Resolution Time Predictor")
request_category = st.selectbox("311 Request Category", list(REQUEST_TYPES.keys()))
request_type = st.selectbox("311 Request Type", REQUEST_TYPES[request_category])
submission_date = st.date_input("Date Submitted", value=date.today())
ward_str = st.selectbox("Ward", WARDS)

if st.button("Predict"):
    ward = int(ward_str.replace("Ward ", ""))
    service_code = REQUEST_TYPES["All Requests"][request_type]
    input_df = pd.DataFrame(
        {"servicecode": [service_code], "ward": [ward], "adddate": [submission_date]}
    )
    input_df["adddate"] = pd.to_datetime(input_df["adddate"])

    processed_df_clf = feature_pipe_clf.transform(input_df)
    processed_df_reg = feature_pipe_reg.transform(input_df)

    # Predicted number of days to resolve request
    preds_num_days = num_days_model.predict(processed_df_reg)

    # Probability that request will take > 21 days to resolve
    preds_21_day = under_21_day_model.predict_proba(processed_df_clf)[:, 1]

    # Probability that request will take < 5 days to resolve
    preds_5_day = under_5_day_model.predict_proba(processed_df_clf)[:, 0]

    st.subheader("Prediction Results")
    st.write(f"Predicted number of days to resolve request: **{preds_num_days[0]}**")
    st.write(f"🟢 Probability resolved in < 5 days: **{preds_5_day[0]:.1%}**")
    st.write(f"🔴 Probability resolved in > 21 days: **{preds_21_day[0]:.1%}**")
