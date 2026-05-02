# =============================================================================
# main.py
# Orchestrates the full inference pipeline at runtime.
# Called by streamlit_app.py when the user clicks Run Analysis.
# Loads a separate LightGBM model per ticker.
# Does not train models. Does not display anything.
# =============================================================================

import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_root, "src")

for _p in [_root, _src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
import pandas as pd

from extractor  import MarketExtractor, NewsExtractor
from processor  import FeatureProcessor, SentimentProcessor
from model      import DirectionModel
from forecaster import PriceForecaster
from optimiser  import PortfolioOptimiser
from database   import Database
from settings   import (
    TICKERS,
    DATE_CONFIG,
    MODELS_DIR,
    PREDICTION_THRESHOLD,
    ensure_dirs,
)


# =============================================================================
# Helpers
# =============================================================================

def _load_lgbm(ticker: str) -> DirectionModel:
    path = MODELS_DIR / f"lgbm_{ticker}.pkl"
    m = DirectionModel()
    m.model = joblib.load(path)
    return m


def _load_forecaster(ticker: str) -> PriceForecaster:
    fc = PriceForecaster(ticker=ticker)
    fc.load()
    return fc


# =============================================================================
# Main Orchestration
# =============================================================================

def orchestrate(
    tickers:   list  = TICKERS,
    start:     str   = DATE_CONFIG["start"],
    end:       str   = DATE_CONFIG["end"],
    split:     str   = DATE_CONFIG["split"],
    threshold: float = PREDICTION_THRESHOLD,
) -> dict:

    ensure_dirs()

    # -------------------------------------------------------------------------
    # Step 1 — Fetch
    # -------------------------------------------------------------------------
    market_extractor = MarketExtractor()
    news_extractor   = NewsExtractor()

    price_data = market_extractor.fetch_multiple(
        tickers=tickers,
        start=start,
        end=end,
    )

    news_df = news_extractor.fetch_headlines()

    # -------------------------------------------------------------------------
    # Step 2 — Process
    # -------------------------------------------------------------------------
    feature_proc   = FeatureProcessor()
    sentiment_proc = SentimentProcessor()

    featured_data = {}
    for ticker, raw_df in price_data.items():
        with_features  = feature_proc.create_features(raw_df)
        with_sentiment = sentiment_proc.process(with_features, news_df)
        featured_data[ticker] = with_sentiment

    # -------------------------------------------------------------------------
    # Step 3 — Load per-ticker LightGBM + Predict on Test Split
    # -------------------------------------------------------------------------
    predictions   = {}
    backtests     = {}
    metrics       = {}
    lgbm_signals  = {}
    fi_frames     = []

    for ticker in tickers:
        df = featured_data.get(ticker)
        if df is None:
            continue

        test_df = df[df.index >= split].copy()
        if test_df.empty:
            continue

        try:
            model = _load_lgbm(ticker)
        except FileNotFoundError:
            continue

        predicted = model.predict(test_df, threshold=threshold)
        bt        = model.backtest(predicted)
        m         = model.compute_metrics(bt)

        predictions[ticker]  = predicted
        backtests[ticker]    = bt
        metrics[ticker]      = m
        lgbm_signals[ticker] = int(predicted["signal"].iloc[-1])

        fi = model.feature_importance()
        fi["ticker"] = ticker
        fi_frames.append(fi)

    # Aggregate feature importance across tickers
    if fi_frames:
        feature_importance = (
            pd.concat(fi_frames)
            .groupby("feature")["importance"]
            .mean()
            .reset_index()
            .sort_values("importance", ascending=False)
        )
    else:
        feature_importance = pd.DataFrame(columns=["feature", "importance"])

    # -------------------------------------------------------------------------
    # Step 4 — Load Prophet + Get Trend Signals + Forecasts
    # -------------------------------------------------------------------------
    trend_signals = {}
    forecasts     = {}

    for ticker in tickers:
        try:
            fc = _load_forecaster(ticker)
            trend_signals[ticker] = fc.trend_signal()
            forecasts[ticker]     = fc.forecast()
        except FileNotFoundError:
            trend_signals[ticker] = 0.0
            forecasts[ticker]     = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Step 5 — Portfolio Optimisation
    # -------------------------------------------------------------------------
    optimiser = PortfolioOptimiser()

    portfolio = optimiser.filter_by_signals(
        price_data=price_data,
        lgbm_signals=lgbm_signals,
        trend_signals=trend_signals,
    )

    # -------------------------------------------------------------------------
    # Step 6 — Persist Results
    # -------------------------------------------------------------------------
    db = Database()

    for ticker in metrics:
        db.save_backtest(ticker, metrics[ticker])
        db.save_predictions(ticker, predictions[ticker])

    db.save_portfolio(
        weights=portfolio["weights"],
        metrics=portfolio["metrics"],
    )

    db.close()

    # -------------------------------------------------------------------------
    # Return everything Streamlit needs
    # -------------------------------------------------------------------------
    return {
        "tickers":            tickers,
        "price_data":         price_data,
        "featured_data":      featured_data,
        "predictions":        predictions,
        "backtests":          backtests,
        "metrics":            metrics,
        "lgbm_signals":       lgbm_signals,
        "trend_signals":      trend_signals,
        "forecasts":          forecasts,
        "portfolio":          portfolio,
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    results = orchestrate()

    print("\nLGBM Signals:")
    for ticker, signal in results["lgbm_signals"].items():
        print(f"  {ticker}: {'BUY' if signal == 1 else 'HOLD/SELL'}")

    print("\nPortfolio Weights:")
    for ticker, weight in results["portfolio"]["weights"].items():
        print(f"  {ticker}: {weight:.2%}")

    print("\nPortfolio Metrics:")
    for k, v in results["portfolio"]["metrics"].items():
        print(f"  {k}: {v:.4f}")