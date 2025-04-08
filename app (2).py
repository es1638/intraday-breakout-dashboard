
import streamlit as st
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Ensure this is the first Streamlit command
st.set_page_config(layout="wide")
st.title("📈 Intraday Breakout Prediction Dashboard")

# Auto-refresh every 2 minutes
st_autorefresh(interval=2 * 60 * 1000, key="datarefresh")

# Load model
model = lgb.Booster(model_file="lightgbm_model.txt")

# Load screener output
try:
    screener_df = pd.read_csv("intraday_with_premarket_features.csv")
    tickers = screener_df["ticker"].unique().tolist()
except Exception as e:
    st.error(f"Failed to load screener file: {e}")
    tickers = []

# Feature extraction from live data
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=10).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("Not enough data to compute features")
    return features.iloc[-1]

# Buy signal logic
def generate_buy_signal(ticker, threshold=0.98):
    try:
        features = get_live_features(ticker)
        X = features.values.reshape(1, -1)
        prob = model.predict(X)[0]
        signal = prob >= threshold
        return signal, prob
    except Exception as e:
        return "Error", str(e)

# Show dashboard table
st.subheader("📊 Live Buy Signals")
results = []
for ticker in tickers:
    signal, prob = generate_buy_signal(ticker)
    results.append({
        "Ticker": ticker,
        "Buy Signal": "✅ Buy" if signal is True else ("❌ No" if signal is False else "⚠️ Error"),
        "Probability": f"{prob:.3f}" if isinstance(prob, float) else prob
    })

df_results = pd.DataFrame(results)
st.dataframe(df_results, use_container_width=True)
