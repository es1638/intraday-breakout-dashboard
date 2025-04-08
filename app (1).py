
import streamlit as st
import pandas as pd
import lightgbm as lgb
import yfinance as yf

st.set_page_config(layout="wide")
st.title("📉 Intraday Breakout Prediction Dashboard")

# Load model
model = lgb.Booster(model_file="lightgbm_model.txt")

# Load screener dataset
screener_df = pd.read_csv("intraday_with_premarket_features.csv")

def get_live_features(ticker: str) -> pd.Series:
    static_row = screener_df[screener_df["ticker"] == ticker]
    if static_row.empty:
        raise ValueError(f"No screener data available for {ticker}")
    
    static_row = static_row.sort_values("date", ascending=False).iloc[0]

    data = yf.download(ticker, period="2d", interval="1m", progress=False)
    if data.empty or len(data) < 11:
        raise ValueError(f"Not enough intraday data for {ticker}")

    data.index = pd.to_datetime(data.index)
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = data["Volume"].rolling(window=10).mean()
    data["rolling_volume_ratio"] = data["Volume"] / data["rolling_volume"]

    features = data[["momentum_10min", "price_change_5min", "rolling_volume_ratio"]].dropna()
    if features.empty:
        raise ValueError(f"No valid feature rows found for {ticker}")

    latest = features.iloc[-1]
    latest["premarket_change"] = static_row["premarket_change"]
    latest["open_vs_premarket"] = static_row["open_vs_premarket"]
    return latest

def generate_buy_signal(ticker: str, model, threshold: float):
    try:
        features = get_live_features(ticker)
        X = features[["premarket_change", "open_vs_premarket", "price_change_5min", "momentum_10min", "rolling_volume_ratio"]].values.reshape(1, -1)
        prob = model.predict(X)[0]
        signal = prob >= threshold
        return signal, prob, features
    except Exception as e:
        return "Error", str(e), None

tickers_input = st.text_input("Enter ticker(s) separated by commas (e.g. TSLA, NVDA, AAPL):", "TSLA")
threshold = st.slider("Buy Signal Threshold", min_value=0.90, max_value=1.0, value=0.9761, step=0.0001)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("🔍 Evaluate"):
    results = []
    for ticker in tickers:
        signal, prob, _ = generate_buy_signal(ticker, model, threshold=threshold)
        results.append({"Ticker": ticker, "Buy Signal": signal, "Probability": prob})
    st.dataframe(pd.DataFrame(results))
