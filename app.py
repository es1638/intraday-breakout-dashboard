import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
import time
from datetime import datetime

# Password Protection
def login():
    st.title("🔐 Secure Access")
    password = st.text_input("Enter password to access the dashboard:", type="password")
    if password != "my_secret":
        st.warning("Incorrect password")
        st.stop()

login()

# Feature list used in your model
FEATURES = [
    'premarket_change', 'open_vs_premarket', 'volume_spike_ratio',
    'price_change_5min', 'momentum_10min', 'rolling_volume_ratio', 'above_vwap'
]

# Load pre-trained LightGBM model
@st.cache_resource
def load_model():
    model = lgb.Booster(model_file='lightgbm_model.txt')
    return model

model = load_model()

# Function to fetch current intraday data
def get_live_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(interval="1m", period="1d", prepost=False)
    return data

# Feature engineering (must match training logic)
def engineer_features(data, static_row):
    data = data.copy()
    data['price_change_5min'] = data['Close'].pct_change(periods=5)
    data['momentum_10min'] = data['Close'].pct_change(periods=10)
    data['rolling_volume'] = data['Volume'].rolling(10).mean()
    data['rolling_volume_ratio'] = data['Volume'] / data['rolling_volume']
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['above_vwap'] = (data['Close'] > data['vwap']).astype(int)

    # Inject static premarket features
    data['premarket_change'] = static_row['premarket_change']
    data['open_vs_premarket'] = static_row['open_vs_premarket']
    data['volume_spike_ratio'] = static_row['volume_spike_ratio']

    return data

# UI Layout
st.set_page_config(layout="wide")
st.title("🚨 Intraday Breakout Prediction Dashboard")

# Input ticker list
st.sidebar.header("Settings")
ticker_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,AMD,NVDA")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Mock static context (you can replace with real file or DB lookup)
@st.cache_data
def get_mock_static(ticker):
    return {
        'premarket_change': np.random.uniform(-0.02, 0.05),
        'open_vs_premarket': np.random.uniform(-0.01, 0.03),
        'volume_spike_ratio': np.random.uniform(0.8, 2.5)
    }

# Threshold from your analysis
THRESHOLD = 0.9761

# Live predictions
results = []
for ticker in tickers:
    try:
        live_data = get_live_data(ticker)
        static_row = get_mock_static(ticker)
        features_df = engineer_features(live_data, static_row)
        latest = features_df.dropna().tail(1)
        if latest.empty:
            continue
        X_live = latest[FEATURES]
        prob = model.predict(X_live)[0]
        results.append({
            'Ticker': ticker,
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Breakout Probability': round(prob, 4),
            'Signal': "✅ BUY" if prob >= THRESHOLD else ""
        })
    except Exception as e:
        results.append({
            'Ticker': ticker,
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Breakout Probability': 'Error',
            'Signal': f"Error: {str(e)}"
        })

# Display results
df_results = pd.DataFrame(results)
st.dataframe(df_results, use_container_width=True)

# Auto-refresh every 45 seconds
st.experimental_rerun()
time.sleep(45)
