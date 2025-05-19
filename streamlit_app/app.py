"""
Develop Streamlit app
"""

from datetime import date
import json

import joblib
import pandas as pd
import streamlist as st


under_21_day_model = joblib.load("../models/under_21_day_model.joblib")
under_5_day_model = joblib.load("../models/under_21_day_model.joblib")

with open("request_categories.json") as f:
    REQUEST_TYPES = json.load(f)
WARDS = [f"Ward {i}" for i in range(1, 9)]

# App UI
st.title("DC 311 Request Resolution Time Predictor")

request_type = st.selectbox("Type of 311 Request", REQUEST_TYPES)
submission_date = st.date_input("Date Submitted", value=date.today())
ward = st.selectbox("Ward", WARDS)

if st.button("Predict"):
    input_df = pd.DataFrame(
        {"servicecode": [request_type], "ward": [ward], "adddate": [submission_date]}
    )

    # TODO - Add inference pipeline for dataframe

    # Probability that request will take > 21 days to resolve
    preds_21_day = under_21_day_model.predict_proba(input_df)[1]

    # Probability that request will take < 5 days to resolve
    preds_5_day = under_5_day_model.predict_proba(input_df)[0]

    st.subheader("Prediction Results")
    st.write(f"🟢 Probability resolved in < 5 days: **{preds_5_day[0]:.1%}**")
    st.write(f"🔴 Probability resolved in > 21 days: **{preds_21_day[0]:.1%}**")
