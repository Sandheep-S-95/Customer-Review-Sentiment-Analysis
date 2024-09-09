import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    try:
        if uploaded_file is not None:
            file = {"file": uploaded_file}
            response = requests.post(prediction_endpoint, files=file)
            if response.status_code == 200:
                response_bytes = BytesIO(response.content)
                response_df = pd.read_csv(response_bytes)
                st.write(response_df)  # Display the predictions
                st.download_button(
                    label="Download Predictions",
                    data=response_bytes,
                    file_name="Predictions.csv",
                    key="result_download_button",
                )
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        else:
            response = requests.post(prediction_endpoint, json={"text": user_input})
            if response.status_code == 200:
                response_data = response.json()
                st.write(f"Predicted sentiment: {response_data['prediction']}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")