import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from streamlit_js_eval import streamlit_js_eval  # New Import

# 1. THEME & VISUAL SETUP
st.set_page_config(page_title="Market Tracker", layout="wide")

# --- 60-SECOND LIVE REFRESH ---
st_autorefresh(interval=60000, limit=100, key="finticker_refresh")

# GET USER TIMEZONE via JavaScript
user_tz = streamlit_js_eval(js_expressions='Intl.DateTimeFormat().resolvedOptions().timeZone', key='tz')

st.markdown("""
    <style>
    .stApp { background-color: #1a1c23; color: #ffffff; } 
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    [data-testid="stMetric"] { 
        background-color: #262931; padding: 20px; border-radius: 10px; border: 1px solid #3f444e;
    }
    [data-testid="stMetricValue"] { color: #4ade80 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Market Tracker")

# 2. SIDEBAR CONTROLS
st.sidebar.header("Terminal Controls")
ticker_input = st.sidebar.text_input("Ticker Symbol", "NVDA").upper()
ma_window = st.sidebar.slider("Analysis Window (MA & Bollinger)", 5, 100, 20)
view_options = ["1 Day (1m)", "5 Days (10m)", "1 Month (1d)", "6 Months (1d)", "YTD (1d)", "1 Year (1d)", "5 Year (1wk)"]
timeframe = st.sidebar.selectbox("Select View", view_options)

# 3. DATA FETCHING
interval_map = {"1 Day (1m)": "1m", "5 Days (10m)": "5m", "1 Month (1d)": "1d", "6 Months (1d)": "1d", "YTD (1d)": "1d", "1 Year (1d)": "1d", "5 Year (1wk)": "1wk"}
period_map = {"1 Day (1m)": "1d", "5 Days (10m)": "5d", "1 Month (1d)": "1mo", "6 Months (1d)": "6mo", "YTD (1d)": "ytd", "1 Year (1d)": "1y", "5 Year (1wk)": "max"}

def load_live_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=['Close'])
    except: return None

raw_data = load_live_data(ticker_input, period_map[timeframe], interval_map[timeframe])

if raw_data is not None and len(raw_data) > 2:
    df = raw_data.copy()
    
    # --- TIMEZONE LOCALIZATION ---
    # Convert UTC to User's Local Timezone
    if user_tz:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(user_tz)
    
    # --- 4. DATA PROCESSING ---
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
    df['Std'] = df['Close'].rolling(window=ma_window).std()
    df['Upper'] = df['MA'] + (df['Std'] * 2)
    df['Lower'] = df['MA'] - (df['Std'] * 2)
    df['Daily_Return'] = df['Close'].pct_change()
    
    if "1 Day" in timeframe:
        df['Display_Date'] = df.index.strftime('%H:%M')
        step = max(1, len(df) // 10)
    elif "5 Days" in timeframe:
        df = df.iloc[::2] 
        df['Display_Date'] = df.index.strftime('%b %d, %H:%M')
        step = max(1, len(df) // 8)
    else:
        df['Display_Date'] = df.index.strftime('%Y-%m-%d')
        step = max(1, len(df) // 8)
    
    # ... [REST OF YOUR METRICS AND CHART CODE REMAINS THE SAME] ...
    latest_price = float(df['Close'].iloc[-1])
    returns = df['Daily_Return'].dropna()
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0
    sharpe = (returns.mean() * 252 - 0.02) / volatility if volatility != 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Live Price", f"${latest_price:,.2f}")
    m2.metric("Period High", f"${df['High'].max():,.2f}")
    m3.metric("Period Low", f"${df['Low'].min():,.2f}")
    m4.metric("Market Risk", "High" if volatility > 0.3 else "Low", f"{volatility:.1%} Vol")
    
    eff_label = "Negative" if sharpe < 0 else "Low" if sharpe < 1 else "High"
    m5.metric("Efficiency", eff_label, f"{sharpe:.2f} Sharpe", delta_color="inverse" if sharpe < 0 else "normal")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df['Display_Date'], y=df['Upper'], line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Display_Date'], y=df['Lower'], line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.07)', name='Volatility Band'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Display_Date'], y=df['Close'], name='Price', line=dict(color='#4ade80', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Display_Date'], y=df['MA'], name='Mean (SMA)', line=dict(color='#fbbf24', dash='dash')), row=1, col=1)
    
    vol_colors = ['#4ade80' if row['Open'] < row['Close'] else '#f87171' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Display_Date'], y=df['Volume'], name='Volume', marker_color=vol_colors, opacity=0.5), row=2, col=1)

    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikecolor='#9ca3af', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikecolor='#9ca3af', spikethickness=1)

    fig.update_layout(
        template="plotly_dark", height=600, margin=dict(l=0, r=180, t=10, b=0),
        hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis2=dict(type='category', tickvals=df['Display_Date'][::step], ticktext=df['Display_Date'][::step], showgrid=False),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)') 
    )
    fig.update_yaxes(side='right', showgrid=True, gridcolor='#3f444e')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Returns Distribution Profile")
    max_range = max(abs(returns.min()), abs(returns.max())) * 1.1
    fig_hist = px.histogram(returns, nbins=50, template="plotly_dark", color_discrete_sequence=['#60a5fa'])
    fig_hist.update_layout(bargap=0.1, xaxis_title="VALUE", yaxis_title="COUNT", xaxis=dict(range=[-max_range, max_range]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.sidebar.download_button(label="📥 Download Data (.csv)", data=df.to_csv().encode('utf-8'), file_name=f'{ticker_input}_data.csv', mime='text/csv')
else:
    st.error(f"Ticker '{ticker_input}' not found or loading...")