
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Load trained model
model = joblib.load("lightgbm_model_converted.pkl")

@st.cache_data(ttl=86400)
def run_daily_screen():
    try:
        from yahoo_fin import stock_info as si
        sp500_tickers = si.tickers_sp500()
        qualified = []
        skipped = 0

        for ticker in sp500_tickers:
            try:
                info = si.get_quote_table(ticker, dict_result=True)
                hist = yf.download(ticker, period="6mo")
                if hist.empty or len(hist) < 60:
                    skipped += 1
                    continue

                avg_volume = hist["Volume"].tail(30).mean()
                current_close = hist["Close"].iloc[-1]
                high_52w = hist["High"].rolling(window=252).max().iloc[-1]

                if avg_volume > 1e7 and info.get("Beta", 0) > 0 and current_close >= 0.6 * high_52w:
                    qualified.append(ticker)
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
                continue

        return qualified, skipped
    except Exception as e:
        st.error(f"Screening failed: {e}")
        return [], 0

@st.cache_data(ttl=120)
def get_live_features(ticker):
    try:
        data = yf.download(ticker, period="2d", interval="1m")
        data.index = pd.to_datetime(data.index)
        volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]

        data["momentum_10min"] = data["Close"].pct_change(periods=10)
        data["price_change_5min"] = data["Close"].pct_change(periods=5)
        data["rolling_volume"] = volume_series.rolling(window=5).mean()
        data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]

        features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
        return features.iloc[-1:]
    except Exception as e:
        raise ValueError(f"Feature generation failed for {ticker}: {e}")

# --- User Interface ---
if "screened_tickers" not in st.session_state:
    st.session_state.screened_tickers, st.session_state.skipped = run_daily_screen()

if st.button("🔁 Refresh Daily Screen"):
    st.session_state.screened_tickers, st.session_state.skipped = run_daily_screen()

st.subheader("Buy Signal Threshold")
threshold = st.slider("", min_value=0.90, max_value=1.0, value=0.98, step=0.01)

st.write(f"✅ Screened tickers: {len(st.session_state.screened_tickers)}")
st.write(f"🚫 Skipped tickers: {st.session_state.skipped}")

if len(st.session_state.screened_tickers) == 0:
    st.warning("Please run the daily screen to populate tickers.")
else:
    st.subheader(f"Evaluating {len(st.session_state.screened_tickers)} Screened Stocks")

    rows = []
    for ticker in st.session_state.screened_tickers:
        try:
            X = get_live_features(ticker)
            if X.empty:
                raise ValueError("No recent features to evaluate.")
            y_pred = model.predict_proba(X)[:, 1]  # Use model's predict_proba
            rows.append({
                "Ticker": ticker,
                "Buy Signal": "✅ Buy" if y_pred[0] >= threshold else "❌ Hold",
                "Probability": y_pred[0]
            })
        except Exception as e:
            rows.append({
                "Ticker": ticker,
                "Buy Signal": "⚠️ Error",
                "Probability": str(e)
            })

    st.dataframe(pd.DataFrame(rows))
    
