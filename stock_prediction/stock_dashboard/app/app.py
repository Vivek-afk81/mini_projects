# =============================================================================
# streamlit_app.py
# UI layer only. Calls main.orchestrate() and displays results.
# No model logic. No data fetching. No calculations.
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import main
from database import Database
from settings import TICKERS, DATE_CONFIG, PREDICTION_THRESHOLD, UI_CONFIG

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] {
        background-color: #0d0d0d;
    }
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #161616;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #f0f0f0;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.75rem;
        margin-top: 2rem;
    }

    /* Signal badge */
    .signal-buy {
        background: #0d2e1a;
        color: #2ecc71;
        border: 1px solid #1a5c35;
        padding: 0.2rem 0.7rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    .signal-sell {
        background: #2e0d0d;
        color: #e74c3c;
        border: 1px solid #5c1a1a;
        padding: 0.2rem 0.7rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }

    /* Divider */
    hr {
        border-color: #222 !important;
    }

    /* Plotly chart bg */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOT THEME
# =============================================================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#888", size=11),
    margin=dict(l=0, r=0, t=36, b=0),
    xaxis=dict(
        gridcolor="#1a1a1a",
        showline=False,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#1a1a1a",
        showline=False,
        zeroline=False,
    ),
)

COLORS = {
    "blue":      "#3a8fd4",
    "green":     "#2ecc71",
    "red":       "#e74c3c",
    "muted":     "#444",
    "highlight": "#f0c040",
}

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.divider()

    selected_tickers = st.multiselect(
        "Tickers",
        options=TICKERS,
        default=TICKERS,
    )

    st.markdown("")
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime(DATE_CONFIG["start"]),
    )
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime(DATE_CONFIG["end"]),
    )
    split_date = st.date_input(
        "Train / Test Split",
        value=pd.to_datetime(DATE_CONFIG["split"]),
    )

    st.markdown("")
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.30,
        max_value=0.70,
        value=PREDICTION_THRESHOLD,
        step=0.01,
        help="Minimum model confidence to generate a BUY signal.",
    )

    st.divider()
    run = st.button("▶  Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        "<div style='color:#333; font-size:0.7rem; text-align:center;'>"
        "Data: Yahoo Finance · RSS Feeds<br>"
        "Models: LightGBM · Prophet<br>"
        "Optimiser: PyPortfolioOpt"
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# HEADER
# =============================================================================

st.markdown("# 📈 Stock ML Dashboard")
st.markdown(
    "<p style='color:#555; margin-top:-0.75rem;'>"
    "LightGBM direction signals · Prophet trend forecasting · "
    "Mean-variance portfolio optimisation"
    "</p>",
    unsafe_allow_html=True,
)

if not run:
    st.info("Configure the settings in the sidebar and click **Run Analysis** to start.")
    st.stop()

if not selected_tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

# =============================================================================
# RUN PIPELINE
# =============================================================================

with st.spinner("Running pipeline — fetching, processing, predicting, optimising..."):
    try:
        results = main.orchestrate(
            tickers=selected_tickers,
            start=str(start_date),
            end=str(end_date),
            split=str(split_date),
            threshold=threshold,
        )
    except FileNotFoundError:
        st.error(
            "Trained models not found. "
            "Please run `python train.py` from your terminal first, then reload."
        )
        st.stop()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# =============================================================================
# SECTION 1 — SIGNALS OVERVIEW
# =============================================================================

st.markdown('<p class="section-header">Current Signals</p>', unsafe_allow_html=True)

sig_cols = st.columns(len(selected_tickers))

for i, ticker in enumerate(selected_tickers):
    signal = results["lgbm_signals"].get(ticker, 0)
    trend  = results["trend_signals"].get(ticker, 0.0)
    m      = results["metrics"].get(ticker, {})

    badge = (
        '<span class="signal-buy">BUY</span>'
        if signal == 1
        else '<span class="signal-sell">HOLD / SELL</span>'
    )
    trend_str = f"{'↑' if trend >= 0 else '↓'} {abs(trend):.1%} 30d trend"

    with sig_cols[i]:
        st.markdown(
            f"**{ticker}** &nbsp; {badge} &nbsp;"
            f"<span style='color:#555; font-size:0.8rem;'>{trend_str}</span>",
            unsafe_allow_html=True,
        )
        ret = m.get("total_return", 0)
        bh  = m.get("buy_hold", 0)
        st.metric(
            label="Strategy Return",
            value=f"{ret:.1%}",
            delta=f"{ret - bh:+.1%} vs B&H",
        )

st.divider()

# =============================================================================
# SECTION 2 — TICKER DEEP DIVE (tabs)
# =============================================================================

st.markdown('<p class="section-header">Ticker Analysis</p>', unsafe_allow_html=True)

valid_tickers = [t for t in selected_tickers if t in results["backtests"]]

if valid_tickers:
    tabs = st.tabs(valid_tickers)

    for tab, ticker in zip(tabs, valid_tickers):
        with tab:
            bt = results["backtests"][ticker]
            m  = results["metrics"][ticker]

            # --- Metrics Row ---
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Total Return",  f"{m['total_return']:.1%}",
                      f"{m['total_return'] - m['buy_hold']:+.1%} vs B&H")
            c2.metric("Sharpe Ratio",  f"{m['sharpe']:.3f}")
            c3.metric("Max Drawdown",  f"{m['max_drawdown']:.1%}")
            c4.metric("Win Rate",      f"{m['win_rate']:.1%}")
            c5.metric("Accuracy",      f"{m['accuracy']:.1%}")
            c6.metric("ROC-AUC",       f"{m['roc_auc']:.3f}")

            # --- Equity Curve ---
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=bt.index, y=bt["equity_curve"],
                name="ML Strategy",
                line=dict(color=COLORS["blue"], width=2),
            ))
            fig_eq.add_trace(go.Scatter(
                x=bt.index, y=bt["buy_hold"],
                name="Buy & Hold",
                line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
            ))
            fig_eq.update_layout(
                title=f"{ticker} — Equity Curve (Test Period)",
                height=340,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                ),
                **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # --- Drawdown ---
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt.index, y=bt["drawdown"],
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.12)",
                line=dict(color=COLORS["red"], width=1),
                name="Drawdown",
            ))
            fig_dd.update_layout(
                title="Drawdown",
                height=180,
                yaxis_tickformat=".0%",
                showlegend=False,
                **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # --- Prophet Forecast ---
            fc = results["forecasts"].get(ticker, pd.DataFrame())
            if not fc.empty:
                st.markdown(
                    '<p class="section-header">Prophet 30-Day Forecast</p>',
                    unsafe_allow_html=True,
                )
                raw_df = results["price_data"][ticker]

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=raw_df.index,
                    y=raw_df["Close"],
                    name="Historical",
                    line=dict(color=COLORS["muted"], width=1.5),
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc.index,
                    y=fc["predicted_price"],
                    name="Forecast",
                    line=dict(color=COLORS["highlight"], width=2),
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc.index.tolist() + fc.index.tolist()[::-1],
                    y=fc["upper_bound"].tolist() + fc["lower_bound"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(240,192,64,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Confidence Interval",
                    showlegend=True,
                ))
                fig_fc.update_layout(
                    title=f"{ticker} — Price Forecast",
                    height=320,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                    ),
                    **PLOT_LAYOUT,
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            # --- Recent Predictions Table ---
            with st.expander("📋 Recent Predictions (last 20 days)"):
                pred_df = results["predictions"][ticker]
                display = pred_df[["signal", "prob", "future_return", "target"]].tail(20).copy()
                display.columns = ["Signal", "Probability", "Actual Return", "Target"]
                display["Probability"]    = display["Probability"].map("{:.2%}".format)
                display["Actual Return"]  = display["Actual Return"].map("{:.2%}".format)
                st.dataframe(display, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 3 — PORTFOLIO OPTIMISATION
# =============================================================================

st.markdown('<p class="section-header">Portfolio Optimisation</p>', unsafe_allow_html=True)

portfolio = results["portfolio"]
weights   = portfolio["weights"]
p_metrics = portfolio["metrics"]

col_left, col_right = st.columns([1, 1])

with col_left:
    pm1, pm2, pm3 = st.columns(3)
    pm1.metric(
        "Expected Return",
        f"{p_metrics.get('expected_annual_return', 0):.1%}",
    )
    pm2.metric(
        "Volatility",
        f"{p_metrics.get('annual_volatility', 0):.1%}",
    )
    pm3.metric(
        "Portfolio Sharpe",
        f"{p_metrics.get('sharpe_ratio', 0):.3f}",
    )

    # Weights bar chart
    weights_df = (
        pd.DataFrame
        .from_dict(weights, orient="index", columns=["Weight"])
        .sort_values("Weight", ascending=True)
        .reset_index()
        .rename(columns={"index": "Ticker"})
    )

    fig_w = px.bar(
        weights_df,
        x="Weight", y="Ticker",
        orientation="h",
        color="Weight",
        color_continuous_scale=["#1a2e3a", COLORS["blue"]],
        text=weights_df["Weight"].map("{:.1%}".format),
    )
    fig_w.update_layout(
        title="Optimal Allocation",
        height=300,
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
        **PLOT_LAYOUT,
    )
    fig_w.update_traces(textposition="outside")
    st.plotly_chart(fig_w, use_container_width=True)

with col_right:
    # Pie chart
    non_zero = {k: v for k, v in weights.items() if v > 0.001}

    if non_zero:
        fig_pie = go.Figure(go.Pie(
            labels=list(non_zero.keys()),
            values=list(non_zero.values()),
            hole=0.55,
            marker=dict(
                colors=[
                    COLORS["blue"], COLORS["green"],
                    COLORS["highlight"], COLORS["red"], "#9b59b6",
                ][:len(non_zero)],
                line=dict(color="#0d0d0d", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#ccc"),
        ))
        fig_pie.update_layout(
            title="Weight Distribution",
            height=340,
            showlegend=False,
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 4 — FEATURE IMPORTANCE
# =============================================================================

st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)

fi = results["feature_importance"]

fig_fi = px.bar(
    fi.sort_values("importance", ascending=True),
    x="importance", y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale=["#1a2e3a", COLORS["blue"]],
)
fig_fi.update_layout(
    title="LightGBM Feature Importance",
    height=420,
    coloraxis_showscale=False,
    xaxis_title="Importance Score",
    yaxis_title="",
    **PLOT_LAYOUT,
)
st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 5 — HISTORICAL BACKTEST LOG (from database)
# =============================================================================

st.markdown('<p class="section-header">Backtest History</p>', unsafe_allow_html=True)

try:
    db = Database()
    history_frames = []

    for ticker in selected_tickers:
        h = db.get_backtest_history(ticker)
        if not h.empty:
            history_frames.append(h)

    db.close()

    if history_frames:
        history_df = pd.concat(history_frames, ignore_index=True)
        history_df = history_df[[
            "ticker", "run_date", "total_return",
            "sharpe", "max_drawdown", "accuracy",
        ]].copy()
        history_df.columns = [
            "Ticker", "Date", "Total Return",
            "Sharpe", "Max Drawdown", "Accuracy",
        ]
        history_df["Total Return"] = history_df["Total Return"].map("{:.2%}".format)
        history_df["Max Drawdown"] = history_df["Max Drawdown"].map("{:.2%}".format)
        history_df["Accuracy"]     = history_df["Accuracy"].map("{:.2%}".format)
        history_df["Sharpe"]       = history_df["Sharpe"].map("{:.3f}".format)

        st.dataframe(history_df, use_container_width=True)
    else:
        st.caption("No history yet. Run the analysis a few times to build a log.")

except Exception:
    st.caption("Database not available on this deployment.")

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    "<div style='text-align:center; color:#333; font-size:0.7rem;'>"
    "Models: LightGBM (direction) · Prophet (trend) · "
    "PyPortfolioOpt (allocation) · "
    "Data: Yahoo Finance · VADER Sentiment via RSS"
    "</div>",
    unsafe_allow_html=True,
)