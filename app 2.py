
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import lightgbm as lgb
import datetime
import os

# ✅ Set page config FIRST
st.set_page_config(layout="wide")
st.title("📈 Intraday Breakout Prediction Dashboard")

# Load model
@st.cache_resource
def load_model():
    return lgb.Booster(model_file="lightgbm_model.txt")

model = load_model()

# Compute live features
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

# Predict
def generate_buy_signal(ticker: str, model, threshold: float = 0.9761):
    features = get_live_features(ticker)
    X = features.values.reshape(1, -1)
    prob = model.predict(X)[0]
    signal = prob >= threshold
    return signal, prob, features

# Streamlit input
tickers_input = st.text_input("Enter ticker(s) separated by commas (e.g. TSLA, NVDA, AAPL):", "TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.9761, step=0.0001)

# Evaluate
if st.button("🔍 Evaluate"):
    results = []
    for ticker in tickers:
        try:
            signal, prob, feats = generate_buy_signal(ticker, model, threshold)
            results.append({
                "Ticker": ticker,
                "Buy Signal": "✅" if signal else "❌",
                "Probability": f"{prob:.4f}",
                **feats.to_dict()
            })
        except Exception as e:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})
    st.dataframe(pd.DataFrame(results))
