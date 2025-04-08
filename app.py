
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

# Load trained LightGBM model
model = joblib.load("lightgbm_model_converted.pkl")
BUY_SIGNAL_THRESHOLD = 0.9761

# Debuggable screener with logging
def get_screened_stocks():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = sp500['Symbol'].tolist()

    screened = []
    skipped = []
    st.subheader("🔍 Screener Logs")

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            beta = info.get("beta", 0)
            avg_volume = info.get("averageDailyVolume10Day", 0)
            fifty_two_week_high = info.get("fiftyTwoWeekHigh", 0)
            current_price = info.get("regularMarketPrice", 0)
            fifty_two_week_high_date = info.get("fiftyTwoWeekHighDate")

            if not all([beta, avg_volume, fifty_two_week_high, current_price]):
                skipped.append((ticker, "Missing data"))
                continue

            # Grace period for 52-week high hit within last 60 days
            days_since_high = (datetime.now() - datetime.fromtimestamp(fifty_two_week_high_date)).days if fifty_two_week_high_date else None

            passes_beta = beta > 0
            passes_volume = avg_volume > 1e7
            passes_high = days_since_high is not None and days_since_high <= 60

            if passes_beta and passes_volume and passes_high:
                screened.append(ticker)
                st.write(f"✅ {ticker}: beta={beta:.2f}, vol={avg_volume}, days since 52wk high={days_since_high}")
            else:
                skipped.append((ticker, f"Filtered out: beta={beta:.2f}, vol={avg_volume}, days since high={days_since_high}"))

        except Exception as e:
            skipped.append((ticker, f"Error: {str(e)}"))

    st.write("---")
    st.write(f"✅ Screened tickers: {len(screened)}")
    st.write(f"🚫 Skipped tickers: {len(skipped)}")

    return screened

@st.cache_data(ttl=120)
def get_live_features(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="2d", interval="1m")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=5).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("Not enough data to compute features.")
    return features.iloc[-1]

# --- Main App Logic ---
screened_tickers = get_screened_stocks()
threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, BUY_SIGNAL_THRESHOLD, 0.01)
st.write(f"### Evaluating {len(screened_tickers)} Screened Stocks")

if st.button("🔍 Evaluate"):
    results = []
    for ticker in screened_tickers:
        try:
            features = get_live_features(ticker)
            prob = model.predict_proba([features])[0][1]
            signal = "✅ Buy" if prob > threshold else "❌ Hold"
            results.append({"Ticker": ticker, "Buy Signal": signal, "Probability": round(prob, 4)})
        except Exception as e:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})

    st.dataframe(pd.DataFrame(results))
