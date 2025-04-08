import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from datetime import datetime

# Load trained LightGBM model
model = lgb.Booster(model_file="lightgbm_model.txt")
BUY_SIGNAL_THRESHOLD = 0.9761

def get_live_features(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=10).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("Not enough data to compute features for " + ticker)
    return features.iloc[-1]

def generate_buy_signal(ticker: str, model, threshold: float = BUY_SIGNAL_THRESHOLD):
    features = get_live_features(ticker)
    X = features.values.reshape(1, -1)
    prob = model.predict(X)[0]
    signal = prob >= threshold
    return signal, prob, features

# Streamlit app
st.title("📈 Intraday Breakout Dashboard")
ticker_input = st.text_input("Enter stock ticker:", "AAPL")

if st.button("Check Buy Signal"):
    try:
        signal, prob, features = generate_buy_signal(ticker_input.upper(), model)
        st.metric("Buy Signal Probability", f"{prob:.4f}")
        st.write("Live Features:", features)
        if signal:
            st.success("✅ Buy Signal Triggered!")
        else:
            st.info("ℹ️ No Buy Signal Yet.")
    except Exception as e:
        st.error(f"Error: {e}")
