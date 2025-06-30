import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
from datetime import timedelta

# Load model and scaler
model = load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Predictor using LSTM")
st.markdown("Predict next 30 days of stock prices for your favorite company!")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, AMZN, META)", "AAPL")

if st.button("Predict"):

    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')[['Close']]
    df.dropna(inplace=True)

    scaled_data = scaler.transform(df[['Close']])
    last_60 = scaled_data[-60:].reshape(1, -1)[0]

    future_preds = []

    for _ in range(30):
        x = np.array(last_60[-60:]).reshape(1, 60, 1)
        pred = model.predict(x)[0][0]
        future_preds.append(pred)
        last_60 = np.append(last_60, pred)

    predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)

    # Save to CSV
    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predicted_prices.flatten()
    })
    prediction_df.to_csv("predictions.csv", index=False)

    st.subheader(f"📊 Predicted Prices for {ticker} (Next 30 Days)")
    st.dataframe(prediction_df)

    st.line_chart(prediction_df.set_index("Date"))

    st.download_button("⬇️ Download CSV", data=prediction_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
