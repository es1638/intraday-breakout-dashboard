
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📈 Intraday Breakout Prediction Dashboard")

# Load model
model = joblib.load("lightgbm_model_converted.pkl")

# Load screener data
try:
    screener_df = pd.read_csv("intraday_with_premarket_features.csv")
except Exception as e:
    st.error(f"Could not load screener file: {e}")
    st.stop()

# Apply predefined conditions
filtered_df = screener_df[
    (screener_df['avg_volume'] > 10_000_000) &
    (screener_df['beta'] > 1) &
    (screener_df['days_since_52wk_high'] <= 10)
]

tickers = filtered_df['ticker'].unique().tolist()

# Feature generator
def get_live_features(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    if data.empty or len(data) < 10:
        raise ValueError("Not enough data for intraday features")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=10).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("No valid features")
    return features.iloc[-1]

# Slider for threshold
threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.9761, step=0.0001)

# Evaluation loop
st.subheader("Evaluating Live Buy Signals")
results = []
for ticker in tickers:
    try:
        features = get_live_features(ticker)
        X = features.values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        signal = "✅ Buy" if prob >= threshold else "❌ Hold"
        results.append({"Ticker": ticker, "Buy Signal": signal, "Probability": round(prob, 4)})
    except Exception as e:
        results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})

st.dataframe(pd.DataFrame(results))
