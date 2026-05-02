# =============================================================================
# train.py
# Runs the full training pipeline once and saves all pickles.
# Run this script locally before launching the Streamlit app.
# Usage: python train.py
# =============================================================================

import sys
from pathlib import Path

# Make sure src/ modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from extractor  import MarketExtractor, NewsExtractor
from processor  import FeatureProcessor, SentimentProcessor
from model      import DirectionModel
from forecaster import PriceForecaster
from database   import Database
from settings   import (
    TICKERS,
    DATE_CONFIG,
    ensure_dirs,
)


def train():
    ensure_dirs()

    print("=" * 60)
    print("STOCK PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1 — Fetch Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Fetching market data...")
    market_extractor = MarketExtractor()
    price_data       = market_extractor.fetch_multiple(
        tickers=TICKERS,
        start=DATE_CONFIG["start"],
        end=DATE_CONFIG["end"],
    )
    print(f"      Fetched {len(price_data)} tickers.")

    print("\n[1/5] Fetching news headlines...")
    news_extractor = NewsExtractor()
    news_df        = news_extractor.fetch_headlines()
    print(f"      Fetched {len(news_df)} headlines.")

    # -------------------------------------------------------------------------
    # Step 2 — Process Features + Sentiment
    # -------------------------------------------------------------------------
    print("\n[2/5] Processing features and sentiment...")
    feature_proc   = FeatureProcessor()
    sentiment_proc = SentimentProcessor()

    featured_data = {}
    for ticker, raw_df in price_data.items():
        with_features   = feature_proc.create_features(raw_df)
        with_sentiment  = sentiment_proc.process(with_features, news_df)
        featured_data[ticker] = with_sentiment
        print(f"      {ticker} — {len(with_sentiment)} rows after processing.")

    # -------------------------------------------------------------------------
    # Step 3 — Train LightGBM per ticker + Walk-Forward Validation
    # -------------------------------------------------------------------------
    print("\n[3/5] Training LightGBM models...")
    db = Database()

    lgbm_model = DirectionModel()

    for ticker, df in featured_data.items():
        print(f"\n      [{ticker}] Walk-forward validation...")
        cv_results = lgbm_model.walk_forward_validate(df)

        print(f"      [{ticker}] Mean Accuracy : {cv_results['mean_accuracy']:.4f}")
        print(f"      [{ticker}] Mean AUC      : {cv_results['mean_auc']:.4f}")
        print(f"      [{ticker}] Std Accuracy  : {cv_results['std_accuracy']:.4f}")

        db.save_cv_results(ticker, cv_results["cv_detail"])

    # Train final model on all tickers combined
    print("\n      Training final LightGBM on all tickers combined...")
    import pandas as pd
    combined_df = pd.concat(featured_data.values(), ignore_index=True)
    lgbm_model.fit(combined_df)
    lgbm_model.save()
    print(f"      Saved → {lgbm_model.save.__code__.co_varnames}")

    from settings import LGBM_MODEL_PATH
    print(f"      Saved → {LGBM_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # Step 4 — Train Prophet per ticker
    # -------------------------------------------------------------------------
    print("\n[4/5] Training Prophet forecasters...")
    for ticker, raw_df in price_data.items():
        print(f"      [{ticker}] Fitting Prophet...")
        forecaster = PriceForecaster(ticker=ticker)
        forecaster.fit(raw_df)
        forecaster.save()

        from settings import prophet_model_path
        print(f"      [{ticker}] Saved → {prophet_model_path(ticker)}")

    # -------------------------------------------------------------------------
    # Step 5 — Run Backtest on Test Split + Save Results
    # -------------------------------------------------------------------------
    print("\n[5/5] Running backtests on test split...")

    from settings import DATE_CONFIG
    split = DATE_CONFIG["split"]

    for ticker, df in featured_data.items():
        test_df = df[df.index >= split].copy()

        if test_df.empty:
            print(f"      [{ticker}] No test data after split date. Skipping.")
            continue

        predicted  = lgbm_model.predict(test_df)
        bt         = lgbm_model.backtest(predicted)
        metrics    = lgbm_model.compute_metrics(bt)

        db.save_backtest(ticker, metrics)
        db.save_predictions(ticker, predicted)

        print(f"      [{ticker}] Total Return : {metrics['total_return']:.2%}")
        print(f"      [{ticker}] Sharpe       : {metrics['sharpe']:.3f}")
        print(f"      [{ticker}] Accuracy     : {metrics['accuracy']:.2%}")

    db.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — all models saved to data/models/")
    print("=" * 60)


if __name__ == "__main__":
    train()