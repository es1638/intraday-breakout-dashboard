# Intraday Breakout Dashboard

This is a real-time Streamlit dashboard that uses a trained LightGBM model to predict whether a stock will experience a 3% breakout from its current price during the trading day.

### Features
- Pulls intraday 1-minute data for selected stocks
- Applies feature engineering consistent with training
- Uses a pre-trained LightGBM model to predict breakout probability
- Displays real-time signals with auto-refresh every 45 seconds

### Deployment
To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

To deploy on [Streamlit Cloud](https://streamlit.io/cloud):
1. Push this folder to a GitHub repo
2. Create a new app on Streamlit Cloud
3. Select `app.py` as the entry point
