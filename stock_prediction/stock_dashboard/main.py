# =============================================================================
# main.py
# Orchestrates the full inference pipeline at runtime.
# Called by streamlit_app.py when the user clicks Run Analysis.
# Does not train models. Does not display anything.
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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
    LGBM_MODEL_PATH,
    PREDICTION_THRESHOLD,
    ensure_dirs,
)


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
    # Step 3 — Load LightGBM + Predict on Test Split
    # -------------------------------------------------------------------------
    lgbm_model = DirectionModel()
    lgbm_model.load(LGBM_MODEL_PATH)

    predictions    = {}
    backtests      = {}
    metrics        = {}
    lgbm_signals   = {}

    for ticker, df in featured_data.items():
        test_df = df[df.index >= split].copy()

        if test_df.empty:
            continue

        predicted = lgbm_model.predict(test_df, threshold=threshold)
        bt        = lgbm_model.backtest(predicted)
        m         = lgbm_model.compute_metrics(bt)

        predictions[ticker]  = predicted
        backtests[ticker]    = bt
        metrics[ticker]      = m

        # Latest signal: 1 = bullish, 0 = bearish
        lgbm_signals[ticker] = int(predicted["signal"].iloc[-1])

    # -------------------------------------------------------------------------
    # Step 4 — Load Prophet + Get Trend Signals
    # -------------------------------------------------------------------------
    trend_signals = {}

    for ticker in tickers:
        try:
            forecaster = PriceForecaster(ticker=ticker)
            forecaster.load()
            trend_signals[ticker] = forecaster.trend_signal()
        except FileNotFoundError:
            # Prophet model not trained yet — skip gracefully
            trend_signals[ticker] = 0.0

    # Prophet forecasts for display
    forecasts = {}
    for ticker in tickers:
        try:
            forecaster = PriceForecaster(ticker=ticker)
            forecaster.load()
            forecasts[ticker] = forecaster.forecast()
        except FileNotFoundError:
            forecasts[ticker] = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Step 5 — Portfolio Optimisation
    # -------------------------------------------------------------------------
    optimiser  = PortfolioOptimiser()

    portfolio  = optimiser.filter_by_signals(
        price_data=price_data,
        lgbm_signals=lgbm_signals,
        trend_signals=trend_signals,
    )

    # -------------------------------------------------------------------------
    # Step 6 — Feature Importance
    # -------------------------------------------------------------------------
    feature_importance = lgbm_model.feature_importance()

    # -------------------------------------------------------------------------
    # Step 7 — Persist Results
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
    # Return everything Streamlit needs in one dict
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