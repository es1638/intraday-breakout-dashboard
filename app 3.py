import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Set page config immediately
st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Auto-refresh every 2 minutes
st_autorefresh(interval=2 * 60 * 1000, key="refresh")

# Load trained model
model = lgb.Booster(model_file="lightgbm_model.txt")

# Load screened tickers from CSV
try:
    screener_df = pd.read_csv("intraday_with_premarket_features.csv")
    screened_tickers = screener_df["ticker"].unique().tolist()
except Exception as e:
    st.error(f"Could not load screener file: {e}")
    screened_tickers = []

# Live feature generator
def get_live_features(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    data.index = pd.to_datetime(data.index)

    if data.empty or len(data) < 10:
        raise ValueError("Not enough data for " + ticker)

    volume_series = data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=10).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]

    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("No valid features for " + ticker)

    return features.iloc[-1]

# Buy signal logic
def generate_buy_signal(ticker: str, threshold: float = 0.98):
    try:
        features = get_live_features(ticker)
        X = features.values.reshape(1, -1)
        prob = model.predict(X)[0]
        signal = prob >= threshold
        return signal, prob, ""
    except Exception as e:
        return False, None, str(e)

# Threshold slider
threshold = st.slider("Buy Signal Threshold", 0.90, 1.0, 0.98, 0.01)

# Evaluate each ticker
results = []
for ticker in screened_tickers:
    signal, prob, error = generate_buy_signal(ticker, threshold)
    results.append({
        "Ticker": ticker,
        "Buy Signal": "✅ Buy" if signal else ("🚫 No" if prob is not None else "⚠️ Error"),
        "Probability": f"{prob:.4f}" if prob is not None else error
    })

# Display results
df_results = pd.DataFrame(results)
st.dataframe(df_results)