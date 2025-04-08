import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
import joblib

# Load trained model
model = joblib.load("lightgbm_model_converted.pkl")
BUY_SIGNAL_THRESHOLD = 0.9761

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Get current day screener results
@st.cache_data(ttl=120)
def get_screened_tickers():
    try:
        df = pd.read_csv("intraday_with_premarket_features.csv")
        # Filter logic
        filtered_df = df[
            (df["average_daily_volume"] > 1e7) &
            (df["beta"] > 1) &
            (df["high_within_10_days"]) &
            (df["date"] == df["date"].max())
        ]
        return filtered_df
    except Exception as e:
        st.error(f"Error loading screener data: {e}")
        return pd.DataFrame()

# Get intraday features for prediction
def get_live_features(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="2d", interval="1m")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    features = pd.Series(dtype=np.float32)
    try:
        features["momentum_10min"] = data["Close"].pct_change(periods=10).iloc[-1]
        features["price_change_5min"] = data["Close"].pct_change(periods=5).iloc[-1]
        features["rolling_volume"] = volume_series.rolling(window=5).mean().iloc[-1]
        features["rolling_volume_ratio"] = volume_series.iloc[-1] / volume_series.rolling(window=30).mean().iloc[-1]
    except Exception:
        raise ValueError("Not enough data to compute features")
    return features

# UI components
threshold = st.slider("Buy Signal Threshold", min_value=0.90, max_value=1.00, value=0.98, step=0.01)

# Screener and Prediction Logic
screened_df = get_screened_tickers()
results = []

if not screened_df.empty:
    for _, row in screened_df.iterrows():
        ticker = row["ticker"]
        try:
            features = get_live_features(ticker)
            proba = model.predict_proba([features])[0][1]
            buy = "✅ Buy" if proba > threshold else "❌ Wait"
            results.append({"Ticker": ticker, "Buy Signal": buy, "Probability": proba})
        except Exception as e:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
else:
    st.info("No tickers met the screening criteria yet today.")