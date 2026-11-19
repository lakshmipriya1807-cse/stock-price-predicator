import streamlit as st, yfinance as yf, numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.title("Stock Price Predictor")
ticker = st.text_input("Ticker", "RELIANCE.NS")  # .NS = NSE, .BO = BSE
period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"])

df = yf.download(ticker, period=period)
df.columns = df.columns.get_level_values(0)  # newer yfinance returns MultiIndex
df["MA20"] = df["Close"].rolling(20).mean()

fig = go.Figure(go.Candlestick(x=df.index, open=df.Open,
                high=df.High, low=df.Low, close=df.Close, name="Price"))
fig.add_scatter(x=df.index, y=df.MA20, name="MA20")
st.plotly_chart(fig, use_container_width=True)

# Naive 7-day forecast
X = np.arange(len(df)).reshape(-1, 1)
model = LinearRegression().fit(X, df["Close"])
future = np.arange(len(df), len(df) + 7).reshape(-1, 1)
st.subheader("Next 7 days (trend estimate)")
st.line_chart(model.predict(future))