
import streamlit as st
import pandas as pd
import lightgbm as lgb
import yfinance as yf

# Must be first Streamlit command
st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Load model
model = lgb.Booster(model_file="lightgbm_model.txt")

# Load screener CSV with premarket features
screener_df = pd.read_csv("intraday_with_premarket_features.csv")

def get_live_features(ticker: str) -> pd.Series:
    static_row = screener_df[screener_df["ticker"] == ticker].sort_values("date", ascending=False).iloc[0]
    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    data.index = pd.to_datetime(data.index)

    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = data["Volume"].rolling(window=10).mean()
    data["rolling_volume_ratio"] = data["Volume"] / data["rolling_volume"]

    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError("Not enough data to compute features for " + ticker)

    latest = features.iloc[-1]
    latest["premarket_change"] = static_row["premarket_change"]
    latest["open_vs_premarket"] = static_row["open_vs_premarket"]
    return latest

def generate_buy_signal(ticker: str, model, threshold: float = 0.9761):
    features = get_live_features(ticker)
    X = features[["premarket_change", "open_vs_premarket", "price_change_5min", "momentum_10min", "rolling_volume_ratio"]].values.reshape(1, -1)
    prob = model.predict(X)[0]
    signal = prob >= threshold
    return signal, prob, features

# UI
tickers_input = st.text_input("Enter ticker(s) separated by commas (e.g. TSLA, NVDA, AAPL):")
threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.9761)

if st.button("🔍 Evaluate"):
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
    results = []
    for ticker in tickers:
        try:
            signal, prob, _ = generate_buy_signal(ticker, model, threshold)
            results.append({"Ticker": ticker, "Buy Signal": "✅ Buy" if signal else "❌ No", "Probability": f"{prob:.4f}"})
        except Exception as e:
            results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})
    st.dataframe(pd.DataFrame(results))
