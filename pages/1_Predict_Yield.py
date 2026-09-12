from pathlib import Path

import streamlit as st

from src.inference import load_artifacts, load_data, make_input_frame, predict_yield


ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
st.title("Crop Yield Predictor")
st.write("Use the trained model to estimate crop yield for a selected scenario.")


@st.cache_data
def get_data():
    return load_data(ROOT)


@st.cache_resource
def get_artifacts():
    return load_artifacts(ROOT)


try:
    yield_data = get_data()
    preprocessor, model, metadata = get_artifacts()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.form("predict-crop-yield"):
    area = st.selectbox("Area", sorted(yield_data["Area"].dropna().unique()))
    item = st.selectbox("Crop", sorted(yield_data["Item"].dropna().unique()))
    year = st.slider(
        "Year",
        int(yield_data["Year"].min()),
        int(yield_data["Year"].max()),
        int(yield_data["Year"].max()),
    )
    rainfall = st.number_input("Average rainfall (mm/year)", min_value=0.0, value=1000.0)
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=100.0)
    temperature = st.number_input("Average temperature (C)", min_value=-20.0, max_value=50.0, value=20.0)
    submitted = st.form_submit_button("Predict yield")

if submitted:
    try:
        inputs = make_input_frame(area, item, year, rainfall, temperature, pesticides)
        estimate = predict_yield(inputs, preprocessor, model, metadata)
        st.success(f"Predicted crop yield: {estimate:,.2f} hg/ha")
    except (TypeError, ValueError) as error:
        st.error(f"Prediction could not be generated: {error}")
