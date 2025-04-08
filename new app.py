
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import time
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📈 Intraday Breakout Prediction Dashboard")

model = joblib.load("lightgbm_model_converted.pkl")

# --- SCREENING LOGIC ---
def run_daily_screen():
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
    selected = []
    st.write("Running daily screener...")
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            beta = info.get("beta", 0)
            vol = info.get("averageVolume", 0)
            high = info.get("fiftyTwoWeekHigh", 0)
            close = info.get("regularMarketPreviousClose", 0)
            hist = yf.download(ticker, period="60d")
            if hist.empty:
                continue
            recent_high = hist['High'].max()
            days_within = (hist['High'].idxmax() - hist.index[-1]).days

            if beta > 0.0 and close >= 0.6 * recent_high and abs(days_within) <= 60:
                selected.append(ticker)
        except:
            continue
    st.success(f"Screened {len(selected)} tickers.")
    return selected

# Button to refresh daily screen
if "screened_tickers" not in st.session_state:
    st.session_state["screened_tickers"] = []

if st.button("🔁 Refresh Daily Screen"):
    st.session_state["screened_tickers"] = run_daily_screen()

# --- LIVE INTRADAY PREDICTION ---
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=10).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    return features.iloc[-1]

st.markdown("---")
threshold = st.slider("Buy Signal Threshold", 0.90, 1.0, 0.9761, step=0.0001)

if st.session_state["screened_tickers"]:
    st.subheader("📊 Live Evaluation for Screened Stocks")
    results = []
    for ticker in st.session_state["screened_tickers"]:
        try:
            features = get_live_features(ticker)
            X = features.values.reshape(1, -1)
            prob = model.predict_proba(X)[0][1]
            signal = "✅ Buy" if prob >= threshold else "❌ Hold"
            results.append({"Ticker": ticker, "Buy Signal": signal, "Probability": round(prob, 4)})
        except Exception as e:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
else:
    st.warning("Please run the daily screen to populate tickers.")
