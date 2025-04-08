
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import joblib

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

model = joblib.load("lightgbm_model_converted.pkl")

# Screener parameters
MIN_AVG_VOLUME = 10_000_000
MIN_BETA = 0.3  # adjusted from 1 to 0.65
DAYS_WITHIN_52WEEK_HIGH = 30  # slightly more grace

@st.cache_data(ttl=120)
def get_screened_stocks():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = sp500['Symbol'].tolist()

    screened = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            if hist.empty or 'volume' not in hist or 'close' not in hist:
                continue

            avg_volume = hist['Volume'].rolling(window=30).mean().iloc[-1]
            beta = info.get('beta', 0)
            high_52w = hist['Close'].max()
            latest_close = hist['Close'].iloc[-1]
            recent_high_date = hist[hist['Close'] == high_52w].index[-1]
            days_since_high = (datetime.now() - recent_high_date.to_pydatetime()).days

            if avg_volume > MIN_AVG_VOLUME and beta > MIN_BETA and days_since_high <= DAYS_WITHIN_52WEEK_HIGH:
                screened.append(ticker)
        except:
            continue

    return screened

def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]

    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=10).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]

    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("Not enough data to compute features")
    return features.iloc[[-1]]

threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.98, step=0.01)

screened_tickers = get_screened_stocks()
st.write(f"Evaluating {len(screened_tickers)} Screened Stocks")

results = []
for ticker in screened_tickers:
    try:
        X_live = get_live_features(ticker)
        prob = model.predict_proba(X_live)[0][1]
        signal = "✅ Buy" if prob >= threshold else "❌ Hold"
    except Exception as e:
        signal = "⚠️ Error"
        prob = str(e)

    results.append({"Ticker": ticker, "Buy Signal": signal, "Probability": prob})

results_df = pd.DataFrame(results)
st.dataframe(results_df)

if st.button("🔍 Evaluate"):
    st.rerun()
