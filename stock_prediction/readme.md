# Stock ML Dashboard

An end-to-end machine learning pipeline for stock direction prediction, price trend forecasting, and portfolio optimisation — deployed as an interactive Streamlit dashboard.

---

## Overview

This project builds a research-grade pipeline that:

- Predicts next-day stock direction using LightGBM with walk-forward validation
- Forecasts 30-day price trends using Facebook Prophet
- Allocates capital across tickers using mean-variance optimisation
- Incorporates news sentiment from free RSS feeds via VADER
- Logs all results to a persistent SQLite database

This is a **research pipeline**, not a live trading system.

---

## Architecture

```
stock_dashboard/
│
├── src/
│   ├── extractor.py       — yfinance market data + RSS news fetching
│   ├── processor.py       — feature engineering + VADER sentiment scoring
│   ├── model.py           — LightGBM classifier + walk-forward CV + backtesting
│   ├── forecaster.py      — Prophet price trend forecasting
│   ├── optimiser.py       — PyPortfolioOpt mean-variance allocation
│   └── database.py        — SQLite persistence layer
│
├── app/
│   └── app.py             — Streamlit dashboard (UI only)
│
├── data/
│   ├── portfolio.db       — SQLite database
│   └── models/            — saved model pickles
│       ├── lgbm_AAPL.pkl
│       └── prophet_AAPL.pkl
│
├── tests/                 — pytest test suite
├── settings.py            — central configuration
├── train.py               — offline training pipeline
├── main.py                — runtime inference orchestration
└── requirements.txt
```

---

## Pipeline Flow

```
train.py  (run once offline)
    │
    ├── MarketExtractor     — fetch OHLCV from Yahoo Finance
    ├── NewsExtractor       — fetch headlines from RSS feeds
    ├── FeatureProcessor    — engineer 15 technical indicators
    ├── SentimentProcessor  — VADER sentiment scoring (pre-split only)
    ├── DirectionModel      — walk-forward CV → fit → save lgbm_{ticker}.pkl
    └── PriceForecaster     — fit Prophet → save prophet_{ticker}.pkl

main.py  (called at runtime by Streamlit)
    │
    ├── Fetch fresh market data + news
    ├── Process features + sentiment
    ├── Load per-ticker LightGBM → predict signals
    ├── Load per-ticker Prophet  → trend signals
    ├── PortfolioOptimiser       → mean-variance weights
    └── Database                 → persist results
```

---

## Features

### Direction Prediction — LightGBM
- Binary classification: predicts whether next-day return is positive
- Per-ticker models — each stock has its own trained classifier
- Walk-forward cross-validation (5 folds) — no data leakage, respects time ordering
- Strict train/test temporal boundary — model never sees future data during training

### Price Forecasting — Prophet
- 30-day ahead price trend forecast per ticker
- Confidence intervals displayed on dashboard
- Trend signal used to adjust expected returns in portfolio optimiser

### Feature Engineering — 15 Features

| Feature | Description |
|---|---|
| `sma_ratio` | SMA 5 / SMA 20 |
| `momentum_10` | 10-day price momentum |
| `ret_mean_5/10` | Rolling mean return |
| `ret_std_5/10` | Rolling return volatility |
| `volume_change` | Day-over-day volume change |
| `volume_ma_ratio` | Volume vs 10-day average |
| `rsi_14` | Relative Strength Index |
| `macd` | EMA 12 - EMA 26 |
| `bb_position` | Position within Bollinger Bands |
| `ret_lag_1/2/3` | Lagged returns |
| `sentiment_score` | Daily VADER compound score from RSS |

### Portfolio Optimisation — PyPortfolioOpt
- Mean-variance optimisation (Markowitz)
- Only allocates to tickers with bullish LightGBM signal
- Prophet trend signal adjusts expected returns before optimisation
- Constraints: max 40% per ticker, min 5% per ticker


---

## Results (Test Period: 2023-2025)

| Ticker | Accuracy | ROC-AUC | Sharpe | Total Return |
|--------|----------|---------|--------|--------------|
| AAPL   | 57.3%    | 0.531   | 1.766  | 76.3%        |
| MSFT   | ~54%     | ~0.52   | ~1.2   | 38.0%        |
| GOOGL  | ~55%     | ~0.53   | ~1.4   | 74.0%        |
| TSLA   | ~54%     | ~0.52   | ~0.9   | 68.2%        |
| NVDA   | ~55%     | ~0.53   | ~1.1   | 217.5%       |

> Strategy underperforms buy-and-hold on most tickers during the 2023-2025 bull market — expected behaviour. Model value is in risk-adjusted returns and drawdown control rather than raw return maximisation.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | yfinance, feedparser |
| ML | LightGBM, scikit-learn |
| Forecasting | Prophet |
| Sentiment | vaderSentiment |
| Optimisation | PyPortfolioOpt |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Database | SQLite |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourname/stock-ml-dashboard.git
cd stock-ml-dashboard/stock_dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train models
```bash
python train.py
```

Takes approximately 5-10 minutes. Trains one LightGBM and one Prophet model per ticker and saves all pickles to `data/models/`.

### 4. Launch dashboard
```bash
streamlit run app/app.py
```

---

## Configuration

All configuration lives in `settings.py`.

```python
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

DATE_CONFIG = {
    "start": "2020-01-01",
    "end":   "2025-01-01",
    "split": "2023-01-01",
}

RISK_PARAMS = {
    "max_weight": 0.40,
    "min_weight": 0.05,
}
```

---

## Design Decisions

**Why walk-forward validation instead of a simple train/test split?**
Standard k-fold cross-validation shuffles data randomly, which allows future information to leak into training. Walk-forward validation trains on expanding windows and tests on the next period — this mirrors how a model would actually be deployed.

**Why per-ticker models instead of one combined model?**
A combined model trained on all tickers simultaneously includes test-period rows from some tickers in the training data of others. Per-ticker models enforce a strict information boundary.

**Why VADER instead of FinBERT?**
VADER runs completely offline with no API key and no cost. For daily sentiment aggregation from headlines it is sufficient and far more practical for a free deployment.

**Why SQLite?**
Zero-cost, zero-configuration, and sufficient for logging backtest results across runs.

---

## Limitations

- Sentiment is generic financial news, not ticker-specific
- Model accuracy (54-58%) is modest — next-day stock direction is near-unpredictable by design
- Not suitable for live trading

---

## License

MIT