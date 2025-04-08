import streamlit as st
import pandas as pd
import lightgbm as lgb
import yfinance as yf
import time
from datetime import datetime

# Auto-refresh every 2 minutes
if "last_ran" not in st.session_state or time.time() - st.session_state.last_ran > 120:
    st.session_state.last_ran = time.time()
    st.rerun()

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Load model
model = lgb.Booster(model_file="lightgbm_model.txt")

# Load screener output
try:
    screener_df = pd.read_csv("intraday_with_premarket_features.csv")
    tickers = screener_df["ticker"].unique().tolist()
except Exception as e:
    st.error("⚠️ Could not load screener file.")
    st.stop()

# Define feature generation
def get_live_features(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    if data.empty or len(data) < 10:
        raise ValueError(f"Not enough data to compute features for {ticker}")
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

# Generate buy signal
def generate_buy_signal(ticker: str, model, threshold: float):
    try:
        features = get_live_features(ticker)
        X = features.values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        signal = prob >= threshold
        return signal, prob, None
    except Exception as e:
        return None, None, str(e)

# Threshold slider
threshold = st.slider("Buy Signal Threshold", min_value=0.90, max_value=1.0, step=0.01, value=0.98)

# Evaluate tickers
results = []
for ticker in tickers:
    signal, prob, error = generate_buy_signal(ticker, model, threshold)
    results.append({
        "Ticker": ticker,
        "Buy Signal": "✅ Yes" if signal else ("❌ No" if signal is not None else "⚠️ Error"),
        "Probability": f"{prob:.3f}" if prob is not None else error or "N/A"
    })

st.dataframe(pd.DataFrame(results))
