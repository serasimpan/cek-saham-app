import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import plotly.graph_objects as go

# Load model dan fitur
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

def prepare_features(data):
    # Validasi data
    if 'Close' not in data.columns or data['Close'].isnull().all():
        raise ValueError("Data tidak memiliki harga penutupan ('Close') yang valid.")
    if 'Volume' not in data.columns or data['Volume'].isnull().all():
        raise ValueError("Data tidak memiliki volume yang valid.")

    data = data.copy()
    data['RSI'] = RSIIndicator(close=data['Close']).rsi()
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd_diff()
    boll = BollingerBands(close=data['Close'])
    data['BB_high'] = boll.bollinger_hband()
    data['BB_low'] = boll.bollinger_lband()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Momentum'] = data['Close'].pct_change(periods=5)
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['Sentiment'] = np.random.uniform(-1, 1, len(data))  # Dummy
    return data.dropna()

def make_predictions(data):
    data = prepare_features(data)
    data['Prediction'] = model.predict(data[features])
    return data

def plot_chart(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    buy = data[data['Prediction'] == 1]
    sell = data[data['Prediction'] == -1]
    hold = data[data['Prediction'] == 0]

    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='BUY',
                             marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='SELL',
                             marker=dict(color='red', size=10, symbol='triangle-down')))
    fig.add_trace(go.Scatter(x=hold.index, y=hold['Close'], mode='markers', name='HOLD',
                             marker=dict(color='blue', size=7, symbol='circle')))

    fig.update_layout(title=f"Live Chart + Signal Prediksi untuk {ticker.upper()}",
                      xaxis_title="Waktu", yaxis_title="Harga",
                      template="plotly_dark", height=600)
    return fig

# Streamlit Web App
st.set_page_config(page_title="Live Stock Chart + Signal", layout="wide")
st.title("📈 Realtime Stock Prediction with Chart")
st.caption("Sinyal intraday dengan ML + chart harga saham")

ticker = st.text_input("Masukkan kode saham (misal: AAPL, MSFT, TSLA)", value="AAPL")

if st.button("Prediksi dan Tampilkan Chart"):
    with st.spinner("Mengambil data dan memproses..."):
        data = yf.download(ticker, period='1d', interval='1m', progress=False)
        if data.empty or len(data) < 30 or data['Close'].isnull().all():
            st.warning("⚠️ Data terlalu sedikit atau tidak memiliki harga penutupan yang valid.")
        else:
            try:
                data = make_predictions(data)
                last_pred = data['Prediction'].iloc[-1]
                signal = "📈 BUY" if last_pred == 1 else "🔻 SELL" if last_pred == -1 else "⏸️ HOLD"
                st.subheader(f"Prediksi Terbaru untuk {ticker.upper()}: {signal}")
                st.plotly_chart(plot_chart(data, ticker), use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi error saat memproses data: {e}")
