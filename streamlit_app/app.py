"""
Develop Streamlit app
"""

from datetime import date
import json
import os.path as osp
import sys

import joblib
import pandas as pd
import streamlit as st


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

under_21_day_model = joblib.load("models/under_21_day_model.joblib")
under_5_day_model = joblib.load("models/under_5_day_model.joblib")
num_days_model = joblib.load("models/num_days_model.joblib")
feature_pipe_clf = joblib.load("models/feature_pipeline_clf.joblib")
feature_pipe_reg = joblib.load("models/feature_pipeline_reg.joblib")

with open("streamlit_app/request_categories.json") as f:
    REQUEST_TYPES = json.load(f)
WARDS = [f"Ward {i}" for i in range(1, 9)]

st.set_page_config(page_title="DC 311 ETA", layout="wide")
with st.sidebar:
    st.header("Enter Your 311 Request Details")
    request_category = st.selectbox(
        "311 Request Category",
        options=list(REQUEST_TYPES.keys()),
        help="Select 'All Requests' if you already know your 311 Request Type",
    )
    request_type = st.selectbox(
        "311 Request Type",
        options=REQUEST_TYPES[request_category],
        help="Start typing in the dropdown to search and filter options",
    )
    submission_date = st.date_input("Date Submitted", value=date.today())
    ward_str = st.selectbox("Your Ward", WARDS)
    submit = st.button("Submit")
    st.markdown("---")
    st.markdown(
        "🔗 [Visit the DC 311 Portal](https://311.dc.gov) to submit or check a 311 request.",
        unsafe_allow_html=False,
    )

st.markdown("# 🕒 DC 311 Requests: How Long Should They Take?")
st.divider()

if not submit:
    st.markdown(
        "#### ⬅️  Use the sidebar to enter your request details, then click *Submit* to get estimated resolution times."
    )
else:
    ward = int(ward_str.replace("Ward ", ""))
    service_code = REQUEST_TYPES["All Requests"][request_type]
    input_df = pd.DataFrame(
        {"servicecode": [service_code], "ward": [ward], "adddate": [submission_date]}
    )
    input_df["adddate"] = pd.to_datetime(input_df["adddate"])

    processed_df_clf = feature_pipe_clf.transform(input_df)
    processed_df_reg = feature_pipe_reg.transform(input_df)

    # Predicted number of days to resolve request
    pred_num_days = num_days_model.predict(processed_df_reg)[0]
    pred_num_days = 0 if pred_num_days < 0 else pred_num_days

    # Probability that request will take > 21 days to resolve
    preds_21_day = under_21_day_model.predict_proba(processed_df_clf)[:, 1]

    # Probability that request will take < 5 days to resolve
    preds_5_day = under_5_day_model.predict_proba(processed_df_clf)[:, 0]

    st.markdown("### Your 311 Resolution Forecast")
    (col1,) = st.columns(1)
    col1.metric(
        label="Estimated resolution time:",
        value=f"{pred_num_days:.0f} days",
    )
    col1, col2 = st.columns(2)
    col1.metric(label="Chance resolved in < 5 days:", value=f"{preds_5_day[0]:.0%}")
    col2.metric(label="Chance resolved in > 21 days", value=f"{preds_21_day[0]:.0%}")
    st.caption(
        "These estimates are based on historical resolution times for similar requests."
    )
