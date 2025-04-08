
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import time
from datetime import datetime, timedelta
import lightgbm as lgb

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Load the trained LightGBM model
model = joblib.load("lightgbm_model_converted.pkl")

# Buy signal threshold
threshold = st.slider("Buy Signal Threshold", min_value=0.90, max_value=1.00, step=0.01, value=0.98)

# Screener conditions
def run_screener():
    sp500 = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
    tickers = sp500["Symbol"].tolist()
    selected = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).info
            hist = yf.download(ticker, period="20d")
            if (
                data.get("averageDailyVolume10Day", 0) > 10_000_000
                and data.get("beta", 0) > 1
                and len(hist) >= 10
                and hist["High"].max() == hist["High"][-10:].max()
            ):
                selected.append(ticker)
        except:
            continue
    return selected

# Get live features
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=5).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    return features.iloc[[-1]] if not features.empty else None

# Run predictions
def evaluate_stocks(tickers):
    results = []
    for ticker in tickers:
        features = get_live_features(ticker)
        if features is None:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": "No features"})
            continue
        try:
            prob = model.predict(features)[0]
            signal = "✅ Buy" if prob >= threshold else "❌ No"
            results.append({"Ticker": ticker, "Buy Signal": signal, "Probability": round(prob, 4)})
        except Exception as e:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})
    return pd.DataFrame(results)

# Screen tickers and cache for 2 minutes
@st.cache_data(ttl=120)
def get_screened_tickers():
    return run_screener()

screened_tickers = get_screened_tickers()
st.subheader(f"Evaluating {len(screened_tickers)} Screened Stocks")

if st.button("🔍 Evaluate"):
    output = evaluate_stocks(screened_tickers)
    st.dataframe(output, use_container_width=True)

