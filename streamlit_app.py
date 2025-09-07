import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

# Load Data
@st.cache_data
def load_data():
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'USDT-USD', 'TRX-USD', 'DOGE-USD']
    currentTimeDate = datetime.now() - relativedelta(months=36)
    start = currentTimeDate.strftime('%Y-%m-%d')
    end = datetime.now().date() + timedelta(days=1)

    final_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        data = data.reset_index()[['Date', 'Close']].rename(columns={'Close': ticker})
        if final_data.empty:
            final_data = data
        else:
            final_data = pd.merge(final_data, data, on='Date', how='outer')

    final_data = final_data.sort_values(by='Date').reset_index(drop=True)
    final_data = final_data.fillna(method='ffill').fillna(method='bfill')
    final_data.set_index('Date', inplace=True)
    return final_data

final_data = load_data()

# Anomaly Detection
def detect_anomalies(df, contamination=0.02):
    df = df.dropna().copy()
    model = IsolationForest(contamination=contamination, random_state=42)
    df['score'] = model.fit_predict(df[['Close']])
    df['anomaly_score'] = model.decision_function(df[['Close']])
    df['is_anomaly'] = df['score'] == -1
    return df

results = {}
for ticker in final_data.columns:
    df = final_data[[ticker]].copy()
    df.columns = ['Close']
    df = detect_anomalies(df)
    df['Ticker'] = str(ticker)

    # Separate BUY/SELL internally
    df['RawSignal'] = np.where(
        (df['is_anomaly']) & (df['Close'].pct_change() < -0.05), 'BUY',
        np.where((df['is_anomaly']) & (df['Close'].pct_change() > 0.10), 'SELL', 'HOLD')
    )

    # For plot: merge BUY/SELL into one category
    df['Signal'] = df['RawSignal'].replace({'BUY':'Buy/Sell', 'SELL':'Buy/Sell'})
    # For table: keep original BUY/SELL
    df['TableSignal'] = df['RawSignal']

    results[ticker] = df

# Streamlit UI
st.set_page_config(page_title="Crypto Anomaly Detection", layout="wide")
st.title("Crypto Anomaly Detection & Recommendations")

# Sidebar
st.sidebar.header("Choose a Ticker")
ticker = st.sidebar.selectbox("Select a cryptocurrency:", list(final_data.columns))

st.sidebar.header("Filter Signals")
signal_filter = st.sidebar.radio(
    "Choose a signal type:",
    ("Anomalies", "Buy/Sell", "HOLD")
)

# Plot Graph
df = results[ticker]
fig = go.Figure()

# Price line
fig.add_trace(go.Scatter(
    x=df.index, y=df['Close'], mode='lines',
    name=f"{ticker} Price", line=dict(color='blue')
))

# Anomaly markers
fig.add_trace(go.Scatter(
    x=df[df['is_anomaly']].index, y=df[df['is_anomaly']]['Close'],
    mode='markers', name="Anomalies",
    marker=dict(color='red', size=8, symbol='x')
))

# BUY/SELL markers
if signal_filter == "Buy/Sell":
    buy_signals = df[df['RawSignal'] == 'BUY']
    sell_signals = df[df['RawSignal'] == 'SELL']

    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Close'],
        mode='markers', name="BUY",
        marker=dict(color="green", size=12, symbol="triangle-up")
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['Close'],
        mode='markers', name="SELL",
        marker=dict(color="red", size=12, symbol="triangle-down")
    ))

# HOLD markers
if signal_filter == "HOLD":
    hold_signals = df[df['RawSignal'] == 'HOLD'].iloc[::10]
    fig.add_trace(go.Scatter(
        x=hold_signals.index, y=hold_signals['Close'],
        mode='markers', name="HOLD",
        marker=dict(color="orange", size=8, symbol="circle")
    ))

fig.update_layout(
    title=f"{ticker} Price with Anomalies & Signals",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    hovermode="x unified",
    width=1000, height=600
)

st.plotly_chart(fig, use_container_width=True)

# Show Table for Buy/Sell and Hold
if signal_filter != "Anomalies":
    st.subheader(f"{ticker} Trading Signals")
    st.dataframe(
        df[df['Signal']==signal_filter][['Close','TableSignal']].rename(columns={'TableSignal':'Signal'}).tail(10)
    )
