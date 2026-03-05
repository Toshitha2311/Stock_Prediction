import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Quant Investment Dashboard", layout="wide")

st.title("📊 Quant Investment Analytics & Recommendation Dashboard")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Configuration")

stock = st.sidebar.text_input("Stock Symbol", "AAPL")
benchmark_symbol = "^GSPC"
period = st.sidebar.selectbox("Timeframe", ["6mo", "1y", "3y", "5y", "max"])
risk_free_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 10.0, 2.0) / 100

# =====================================================
# DATA FETCH
# =====================================================
df = yf.download(stock, period=period)
benchmark = yf.download(benchmark_symbol, period=period)

if df.empty:
    st.error("Invalid Stock Symbol")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if isinstance(benchmark.columns, pd.MultiIndex):
    benchmark.columns = benchmark.columns.get_level_values(0)

df = df.dropna()
benchmark = benchmark.dropna()

# Convert all columns to numeric safely
for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if col in benchmark.columns:
        benchmark[col] = pd.to_numeric(benchmark[col], errors="coerce")

# =====================================================
# RETURNS
# =====================================================
df["Return"] = df["Close"].pct_change()
benchmark["Return"] = benchmark["Close"].pct_change()

combined = pd.concat([df["Return"], benchmark["Return"]], axis=1)
combined.columns = ["Stock", "Market"]
combined = combined.dropna()

trading_days = 252

# =====================================================
# CORE METRICS
# =====================================================
mean_return = float(df["Return"].mean() * trading_days)
volatility = float(df["Return"].std() * np.sqrt(trading_days))

sharpe = float((mean_return - risk_free_rate) / volatility) if volatility != 0 else 0

downside = df[df["Return"] < 0]["Return"]
downside_std = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 0 else 0
sortino = float((mean_return - risk_free_rate) / downside_std) if downside_std != 0 else 0

cov_matrix = np.cov(combined["Stock"], combined["Market"])
beta = float(cov_matrix[0][1] / cov_matrix[1][1]) if cov_matrix[1][1] != 0 else 0

market_return = float(combined["Market"].mean() * trading_days)
alpha = float(mean_return - (risk_free_rate + beta * (market_return - risk_free_rate)))

# CAGR
start_price = float(df["Close"].iloc[0])
end_price = float(df["Close"].iloc[-1])
years = len(df) / trading_days
cagr = float((end_price / start_price) ** (1 / years) - 1) if years > 0 else 0

# Drawdown
df["Cumulative"] = (1 + df["Return"]).cumprod()
df["Peak"] = df["Cumulative"].cummax()
df["Drawdown"] = (df["Cumulative"] - df["Peak"]) / df["Peak"]
max_dd = float(df["Drawdown"].min())

# VaR & CVaR
VaR = float(np.percentile(df["Return"].dropna(), 5))
CVaR = float(df[df["Return"] <= VaR]["Return"].mean())

# =====================================================
# RSI + MA
# =====================================================
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

latest_rsi = float(df["RSI"].iloc[-1])

df["MA50"] = df["Close"].rolling(50).mean()
latest_price = float(df["Close"].iloc[-1])
latest_ma50 = float(df["MA50"].iloc[-1])

# =====================================================
# RULE-BASED SCORING MODEL
# =====================================================
score = 0
if latest_price > latest_ma50:
    score += 1
if sharpe > 1:
    score += 1
if alpha > 0:
    score += 1
if 40 <= latest_rsi <= 70:
    score += 1
if max_dd > -0.30:
    score += 1
if beta < 1.5:
    score += 1

if score >= 5:
    recommendation = "🟢 Strong Buy"
    color = "#00c853"
elif score >= 3:
    recommendation = "🟢 Buy"
    color = "#64dd17"
elif score == 2:
    recommendation = "🟡 Hold"
    color = "#ffb300"
else:
    recommendation = "🔴 Avoid"
    color = "#d50000"

# =====================================================
# KPI DISPLAY
# =====================================================
st.subheader("📌 Risk & Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{cagr*100:.2f}%")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Sortino Ratio", f"{sortino:.2f}")
col4.metric("Beta", f"{beta:.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Alpha", f"{alpha*100:.2f}%")
col6.metric("Volatility", f"{volatility*100:.2f}%")
col7.metric("Max Drawdown", f"{max_dd*100:.2f}%")
col8.metric("VaR (95%)", f"{VaR*100:.2f}%")

st.metric("Conditional VaR (CVaR)", f"{CVaR*100:.2f}%")

# =====================================================
# RECOMMENDATION DISPLAY
# =====================================================
st.subheader("📢 Investment Recommendation")
st.markdown(
    f"""
    <div style="
        padding:25px;
        border-radius:15px;
        text-align:center;
        font-size:28px;
        font-weight:600;
        background-color:{color};
        color:white;">
        {recommendation}
    </div>
    """,
    unsafe_allow_html=True
)
st.write(f"Model Score: {score}/6")

# =====================================================
# PRICE CHART
# =====================================================
st.subheader("📈 Price & MA50")
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(width=2)))
price_fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(dash="dash")))
price_fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(price_fig, width='stretch')

# =====================================================
# ROLLING RISK
# =====================================================
st.subheader("📊 Rolling Volatility")
df["Rolling Volatility"] = df["Return"].rolling(30).std() * np.sqrt(trading_days)
risk_fig = go.Figure()
risk_fig.add_trace(go.Scatter(x=df.index, y=df["Rolling Volatility"], name="Rolling Volatility"))
risk_fig.update_layout(template="plotly_dark", height=350)
st.plotly_chart(risk_fig, width='stretch')

# =====================================================
# RETURN DISTRIBUTION
# =====================================================
st.subheader("📉 Return Distribution")
hist = go.Figure()
hist.add_trace(go.Histogram(x=df["Return"], nbinsx=50))
hist.update_layout(template="plotly_dark", height=350)
st.plotly_chart(hist, width='stretch')

# =====================================================
# STOCK VS MARKET
# =====================================================
st.subheader("📊 Stock vs Market")
comparison = go.Figure()
comparison.add_trace(go.Scatter(x=combined.index, y=(1 + combined["Stock"]).cumprod(), name="Stock"))
comparison.add_trace(go.Scatter(x=combined.index, y=(1 + combined["Market"]).cumprod(), name="S&P 500"))
comparison.update_layout(template="plotly_dark", height=400)
st.plotly_chart(comparison, width='stretch')

# =====================================================
# RAW DATA
# =====================================================
with st.expander("View Raw Data"):
    st.dataframe(df)

# =====================================================
# DOWNLOAD REPORTS
# =====================================================
st.subheader("💾 Download Reports")

# Metrics dataframe
metrics_df = pd.DataFrame({
    "Metric": ["CAGR", "Sharpe", "Sortino", "Beta", "Alpha", "Volatility", "Max Drawdown", "VaR (95%)", "CVaR"],
    "Value": [cagr, sharpe, sortino, beta, alpha, volatility, max_dd, VaR, CVaR]
})

csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
csv_data = df.to_csv(index=True).encode('utf-8')

col1, col2 = st.columns(2)
col1.download_button("📥 Download Metrics CSV", data=csv_metrics, file_name=f"{stock}_metrics.csv", mime="text/csv")
col2.download_button("📥 Download Raw Data CSV", data=csv_data, file_name=f"{stock}_raw_data.csv", mime="text/csv")