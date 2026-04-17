import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock ML Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("lgbm_model.pkl")

model = load_model()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    ticker     = st.text_input("Ticker Symbol", value="AAPL").upper()
    start_date = st.date_input("Start Date",    value=pd.to_datetime("2020-01-01"))
    end_date   = st.date_input("End Date",      value=pd.to_datetime("2025-01-01"))
    split_date = st.date_input("Train/Test Split", value=pd.to_datetime("2023-01-01"))
    threshold  = st.slider("Prediction Threshold", 0.30, 0.70, 0.35, 0.01)
    run        = st.button("▶  Run Analysis", use_container_width=True)

# ── DATA & FEATURE FUNCTIONS ─────────────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    df["return"] = df["Close"].pct_change()
    return df.dropna()

def create_features(df):
    df = df.copy()
    df["sma_5"]            = df["Close"].rolling(5).mean()
    df["sma_10"]           = df["Close"].rolling(10).mean()
    df["sma_20"]           = df["Close"].rolling(20).mean()
    df["sma_ratio"]        = df["sma_5"] / df["sma_20"]
    df["momentum_10"]      = df["Close"] / df["Close"].shift(10) - 1
    df["ret_std_5"]        = df["return"].rolling(5).std()
    df["ret_std_10"]       = df["return"].rolling(10).std()
    df["ret_mean_5"]       = df["return"].rolling(5).mean()
    df["ret_mean_10"]      = df["return"].rolling(10).mean()
    df["volume_change"]    = df["Volume"].pct_change()
    df["volume_ma_ratio"]  = df["Volume"] / df["Volume"].rolling(10).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))
    for lag in range(1, 6):
        df[f"ret_lag_{lag}"] = df["return"].shift(lag)
    df["future_return"] = df["return"].shift(-1)
    df["target"]        = (df["future_return"] > 0).astype(int)
    return df.dropna()

FEATURES = [
    "sma_ratio", "momentum_10",
    "ret_mean_5", "ret_std_5",
    "ret_mean_10", "ret_std_10",
    "volume_change", "volume_ma_ratio",
    "rsi_14",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5"
]

def backtest(df_test, preds, cost=0.001):
    df = df_test.copy()
    df["position"]         = preds
    df["strategy_return"]  = df["position"] * df["future_return"]
    df["trade"]            = df["position"].diff().abs().fillna(0)
    df["net_return"]       = df["strategy_return"] - df["trade"] * cost
    df["equity_curve"]     = (1 + df["net_return"]).cumprod()
    df["buy_hold"]         = (1 + df["return"]).cumprod()
    df["drawdown"]         = df["equity_curve"] / df["equity_curve"].cummax() - 1
    return df

# ── MAIN DASHBOARD ───────────────────────────────────────────────────────────
st.title(f"📈 Stock Direction Prediction Dashboard")

if not run:
    st.info("Configure the settings in the sidebar and click **Run Analysis** to start.")
    st.stop()

# Load + process
with st.spinner(f"Fetching {ticker} data..."):
    raw = load_data(ticker, str(start_date), str(end_date))
    df  = create_features(raw)

# Convert index to timezone-naive for safe comparison
df.index = pd.to_datetime(df.index).tz_localize(None)
split_dt = pd.to_datetime(split_date)

train = df[df.index < split_dt]
test  = df[df.index >= split_dt]

# Guard: catch empty splits and show a clear error
if test.empty:
    st.error(f"No test data found after {split_date}. "
             f"Your data runs from {df.index[0].date()} to {df.index[-1].date()}. "
             f"Move the split date earlier.")
    st.stop()

if train.empty:
    st.error("Training set is empty. Move the split date later.")
    st.stop()

# Verify features exist
missing = [f for f in FEATURES if f not in test.columns]
if missing:
    st.error(f"Missing features in data: {missing}")
    st.stop()

X_test, y_test = test[FEATURES], test["target"]

# Final safety check
if X_test.shape[0] == 0 or X_test.shape[1] == 0:
    st.error("X_test is empty after feature selection. Check your date range.")
    st.stop()

# Predict
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= threshold).astype(int)
bt    = backtest(test, preds)

# ── METRICS ROW ──────────────────────────────────────────────────────────────
total_ret  = bt["equity_curve"].iloc[-1] - 1
sharpe     = bt["net_return"].mean() / bt["net_return"].std() * np.sqrt(252)
max_dd     = bt["drawdown"].min()
win_rate   = (bt["net_return"] > 0).mean()
accuracy   = accuracy_score(y_test, preds)
auc        = roc_auc_score(y_test, probs)
n_trades   = int(bt["trade"].sum())
bh_ret     = bt["buy_hold"].iloc[-1] - 1

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Return",    f"{total_ret:.1%}",  f"{total_ret - bh_ret:+.1%} vs B&H")
col2.metric("Sharpe Ratio",    f"{sharpe:.3f}")
col3.metric("Max Drawdown",    f"{max_dd:.1%}")
col4.metric("Win Rate",        f"{win_rate:.1%}")
col5.metric("Accuracy",        f"{accuracy:.1%}")
col6.metric("ROC-AUC",         f"{auc:.3f}")

st.divider()

# ── EQUITY CURVE ─────────────────────────────────────────────────────────────
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=bt.index, y=bt["equity_curve"],
    name="ML Strategy", line=dict(color="#378ADD", width=2)
))
fig_eq.add_trace(go.Scatter(
    x=bt.index, y=bt["buy_hold"],
    name="Buy & Hold", line=dict(color="#888780", width=2, dash="dot")
))
fig_eq.update_layout(
    title=f"Equity Curve — {ticker} (Test Period)",
    yaxis_title="Portfolio Value (1.0 = start)",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=380, margin=dict(l=0, r=0, t=40, b=0)
)

# ── DRAWDOWN ─────────────────────────────────────────────────────────────────
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=bt.index, y=bt["drawdown"],
    fill="tozeroy", fillcolor="rgba(226,75,74,0.2)",
    line=dict(color="#E24B4A", width=1),
    name="Drawdown"
))
fig_dd.update_layout(
    title="Strategy Drawdown",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    height=200,
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=False
)

st.plotly_chart(fig_eq, use_container_width=True)
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ── FEATURE IMPORTANCE + RETURN DISTRIBUTION ─────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature":    FEATURES,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=True)

        fig_fi = px.bar(
            fi, x="importance", y="feature",
            orientation="h", title="Feature Importance",
            color="importance",
            color_continuous_scale=["#B5D4F4", "#378ADD", "#042C53"]
        )
        fig_fi.update_layout(
            height=420, margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False,
            yaxis_title="", xaxis_title="Importance Score"
        )
        st.plotly_chart(fig_fi, use_container_width=True)

with col_b:
    fig_ret = go.Figure()
    strat_ret = bt["net_return"] * 100
    fig_ret.add_trace(go.Histogram(
        x=strat_ret, nbinsx=40,
        name="Strategy", opacity=0.75,
        marker_color="#378ADD"
    ))
    fig_ret.add_vline(
        x=strat_ret.mean(), line_dash="dash",
        line_color="#E24B4A",
        annotation_text=f"Mean: {strat_ret.mean():.3f}%",
        annotation_position="top right"
    )
    fig_ret.update_layout(
        title="Daily Return Distribution (Strategy)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=420, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    st.plotly_chart(fig_ret, use_container_width=True)

st.divider()

# ── RAW PREDICTION TABLE ──────────────────────────────────────────────────────
with st.expander("📋 View Prediction Data (last 30 days)"):
    display_df = bt[["close" if "close" in bt.columns else "Close",
                      "position", "net_return",
                      "equity_curve", "drawdown"]].tail(30).copy()
    display_df.columns = ["Close", "Position", "Net Return", "Equity", "Drawdown"]
    display_df["Net Return"] = display_df["Net Return"].map("{:.2%}".format)
    display_df["Equity"]     = display_df["Equity"].map("{:.3f}".format)
    display_df["Drawdown"]   = display_df["Drawdown"].map("{:.2%}".format)
    st.dataframe(display_df, use_container_width=True)

st.caption(f"Model: {type(model).__name__} · Threshold: {threshold:.2f} · Trades: {n_trades} · Data: Yahoo Finance")