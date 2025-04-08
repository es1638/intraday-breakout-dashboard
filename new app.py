
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import joblib
from datetime import datetime, timedelta

# Load trained LightGBM model
model = joblib.load("lightgbm_model_converted.pkl")

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# --- Daily Screener ---
@st.cache_data(show_spinner=False)
def get_screened_stocks():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = sp500['Symbol'].tolist()
    screened = []
    skipped = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if hist.empty or len(hist) < 60:
                skipped.append(ticker)
                continue

            hist = hist.dropna()
            avg_volume = hist['Volume'].tail(30).mean()
            if avg_volume < 1e7:
                skipped.append(ticker)
                continue

            close = hist['Close'][-1]
            high_52wk = hist['High'].rolling(window=252, min_periods=1).max().iloc[-1]
            if close < 0.6 * high_52wk:
                skipped.append(ticker)
                continue

            info = stock.info
            beta = info.get('beta', 0)
            if beta is None or beta <= 0:
                skipped.append(ticker)
                continue

            screened.append(ticker)
        except Exception as e:
            skipped.append(ticker)
        time.sleep(0.25)

    return screened, skipped

# --- Feature Generation ---
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    if data.empty or len(data) < 10:
        raise ValueError("Not enough intraday data for " + ticker)

    volume_series = data['Volume']
    features = pd.DataFrame({
        'momentum_10min': data['Close'].pct_change(periods=10),
        'price_change_5min': data['Close'].pct_change(periods=5),
        'rolling_volume': volume_series.rolling(window=10).mean(),
        'rolling_volume_ratio': volume_series.rolling(window=10).mean() / volume_series.rolling(window=30).mean()
    }).dropna()

    return features.iloc[-1:]

# --- Buy Signal Prediction ---
def evaluate_stock(ticker):
    try:
        features = get_live_features(ticker)
        proba = model.predict_proba(features)[0][1]
        return proba
    except Exception as e:
        return str(e)

# --- UI ---
if "screened_tickers" not in st.session_state:
    st.session_state.screened_tickers = []

if st.button("🔄 Refresh Daily Screen"):
    with st.spinner("Running daily screener..."):
        screened, skipped = get_screened_stocks()
        st.session_state.screened_tickers = screened
        st.success(f"Screened {len(screened)} tickers.")

threshold = st.slider("Buy Signal Threshold", min_value=0.90, max_value=1.00, value=0.98, step=0.01)

if st.session_state.screened_tickers:
    results = []
    for ticker in st.session_state.screened_tickers:
        prob = evaluate_stock(ticker)
        if isinstance(prob, float):
            signal = "✅ Buy" if prob >= threshold else "No"
        else:
            signal = "⚠️ Error"
        results.append((ticker, signal, prob))

    df = pd.DataFrame(results, columns=["Ticker", "Buy Signal", "Probability"])
    st.dataframe(df)
else:
    st.warning("Please run the daily screen to populate tickers.")
