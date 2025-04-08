
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import time
from datetime import datetime, timedelta
import lightgbm as lgb
from typing import List

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

BUY_THRESHOLD = st.slider("Buy Signal Threshold", 0.9, 1.0, 0.98, 0.01)

@st.cache_data(show_spinner=False)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()

@st.cache_data(show_spinner=False)
def get_screener_results() -> List[str]:
    tickers = get_sp500_tickers()
    screened = []
    for ticker in tickers:
        try:
            time.sleep(0.25)  # prevent rate limiting
            hist = yf.download(ticker, period="6mo", progress=False)
            if hist.empty or 'Volume' not in hist or 'High' not in hist or 'Close' not in hist:
                st.write(f"⚠️ Skipped {ticker} due to missing data")
                continue

            avg_volume = hist['Volume'].mean()
            high_52wk = hist['High'].max()
            current_price = hist['Close'].iloc[-1]

            if avg_volume > 10_000_000 and current_price >= 0.60 * high_52wk:
                screened.append(ticker)
                st.write(f"✅ {ticker} | AvgVol: {int(avg_volume)} | Close: {current_price:.2f} | 52W High: {high_52wk:.2f}")
            else:
                st.write(f"❌ {ticker} filtered | AvgVol: {int(avg_volume)} | Close: {current_price:.2f} | 52W High: {high_52wk:.2f}")
        except Exception as e:
            st.write(f"🛑 Error with {ticker}: {e}")
            continue
    return screened

st.markdown("### 🧠 Refresh Daily Screen")
if st.button("🔁 Refresh Daily Screen"):
    with st.spinner("Running daily screener..."):
        tickers = get_screener_results()
        st.success(f"Screened {len(tickers)} tickers.")
else:
    st.warning("Please run the daily screen to populate tickers.")

