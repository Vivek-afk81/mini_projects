# Runs the full training pipeline once and saves all pickles.
# Trains a separate LightGBM model per ticker with strict temporal boundary.

import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_root, "src")

for _p in [_root, _src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from extractor  import MarketExtractor, NewsExtractor
from processor  import FeatureProcessor, SentimentProcessor
from model      import DirectionModel
from forecaster import PriceForecaster
from database   import Database
from settings   import (
    TICKERS,
    DATE_CONFIG,
    MODELS_DIR,
    ensure_dirs,
)

import joblib
from pathlib import Path


def lgbm_model_path(ticker: str) -> Path:
    return MODELS_DIR / f"lgbm_{ticker}.pkl"


def train():
    ensure_dirs()

    print("=" * 60)
    print("STOCK PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    split = DATE_CONFIG["split"]

    # -------------------------------------------------------------------------
    # Step 1 — Fetch Market Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Fetching market data...")
    market_extractor = MarketExtractor()
    price_data       = market_extractor.fetch_multiple(
        tickers=TICKERS,
        start=DATE_CONFIG["start"],
        end=DATE_CONFIG["end"],
    )
    print(f"      Fetched {len(price_data)} tickers.")

    # -------------------------------------------------------------------------
    # Step 2 — Fetch News + Score Sentiment
    # Only headlines published BEFORE split date are used in training.
    # This prevents present-day sentiment leaking into historical features.
    # -------------------------------------------------------------------------
    print("\n[2/5] Fetching and scoring news sentiment...")
    news_extractor = NewsExtractor()
    news_df        = news_extractor.fetch_headlines()

    news_df = news_df[news_df["date"] < split].copy()
    print(f"      {len(news_df)} headlines before split date {split}.")

    sentiment_proc = SentimentProcessor()
    feature_proc   = FeatureProcessor()

    # -------------------------------------------------------------------------
    # Step 3 — Process Features Per Ticker
    # -------------------------------------------------------------------------
    print("\n[3/5] Processing features per ticker...")
    featured_data = {}

    for ticker, raw_df in price_data.items():
        with_features  = feature_proc.create_features(raw_df)
        with_sentiment = sentiment_proc.process(with_features, news_df)
        featured_data[ticker] = with_sentiment
        print(f"      {ticker} — {len(with_sentiment)} rows total.")

    # -------------------------------------------------------------------------
    # Step 4 — Train LightGBM Per Ticker
    # Each model is trained strictly on data before split date.
    # Walk-forward validation also runs on training data only.
    # Test metrics are computed on unseen data after split date.
    # -------------------------------------------------------------------------
    print("\n[4/5] Training LightGBM per ticker...")
    db = Database()

    for ticker, df in featured_data.items():

        train_df = df[df.index < split].copy()
        test_df  = df[df.index >= split].copy()

        if train_df.empty:
            print(f"      [{ticker}] No training data before {split}. Skipping.")
            continue

        if test_df.empty:
            print(f"      [{ticker}] No test data after {split}. Skipping.")
            continue

        print(f"\n      [{ticker}] Train rows : {len(train_df)}")
        print(f"      [{ticker}] Test rows  : {len(test_df)}")

        model      = DirectionModel()
        cv_results = model.walk_forward_validate(train_df)

        print(f"      [{ticker}] CV Mean Accuracy : {cv_results['mean_accuracy']:.4f}")
        print(f"      [{ticker}] CV Mean AUC      : {cv_results['mean_auc']:.4f}")
        print(f"      [{ticker}] CV Std Accuracy  : {cv_results['std_accuracy']:.4f}")

        db.save_cv_results(ticker, cv_results["cv_detail"])

        model.fit(train_df)

        path = lgbm_model_path(ticker)
        joblib.dump(model.model, path)
        print(f"      [{ticker}] Saved → {path}")

        predicted = model.predict(test_df)
        bt        = model.backtest(predicted)
        metrics   = model.compute_metrics(bt)

        db.save_backtest(ticker, metrics)
        db.save_predictions(ticker, predicted)

        print(f"      [{ticker}] Test Total Return : {metrics['total_return']:.2%}")
        print(f"      [{ticker}] Test Sharpe       : {metrics['sharpe']:.3f}")
        print(f"      [{ticker}] Test Accuracy     : {metrics['accuracy']:.2%}")
        print(f"      [{ticker}] Test ROC-AUC      : {metrics['roc_auc']:.3f}")

    # -------------------------------------------------------------------------
    # Step 5 — Train Prophet Per Ticker
    # Prophet uses only Close price — no sentiment, no target leakage.
    # -------------------------------------------------------------------------
    print("\n[5/5] Training Prophet forecasters...")

    for ticker, raw_df in price_data.items():
        print(f"      [{ticker}] Fitting Prophet...")
        forecaster = PriceForecaster(ticker=ticker)
        forecaster.fit(raw_df)
        forecaster.save()
        print(f"      [{ticker}] Saved → prophet_{ticker}.pkl")

    db.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — models saved to data/models/")
    print("=" * 60)


if __name__ == "__main__":
    train()