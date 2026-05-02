# =============================================================================
# processor.py
# Responsible for feature engineering and sentiment scoring.
# Expects raw DataFrames from extractor.py.
# Returns fully featured DataFrames ready for model.py and forecaster.py.
# =============================================================================

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from settings import FEATURES


class FeatureProcessor:

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Trend ---
        df["sma_5"]       = df["Close"].rolling(5).mean()
        df["sma_10"]      = df["Close"].rolling(10).mean()
        df["sma_20"]      = df["Close"].rolling(20).mean()
        df["sma_ratio"]   = df["sma_5"] / df["sma_20"]
        df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

        # --- Volatility / Return Stats ---
        df["ret_mean_5"]  = df["return"].rolling(5).mean()
        df["ret_std_5"]   = df["return"].rolling(5).std()
        df["ret_mean_10"] = df["return"].rolling(10).mean()
        df["ret_std_10"]  = df["return"].rolling(10).std()

        # --- Volume ---
        df["volume_change"]   = df["Volume"].pct_change()
        df["volume_ma_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()

        # --- RSI ---
        delta       = df["Close"].diff()
        gain        = delta.clip(lower=0).rolling(14).mean()
        loss        = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + gain / loss))

        # --- MACD ---
        ema12        = df["Close"].ewm(span=12).mean()
        ema26        = df["Close"].ewm(span=26).mean()
        df["macd"]   = ema12 - ema26

        # --- Bollinger Band Position ---
        bb_mid             = df["Close"].rolling(20).mean()
        bb_std             = df["Close"].rolling(20).std()
        df["bb_position"]  = (df["Close"] - bb_mid) / (2 * bb_std)

        # --- Lagged Returns ---
        for lag in range(1, 4):
            df[f"ret_lag_{lag}"] = df["return"].shift(lag)

        # --- Target ---
        df["future_return"] = df["return"].shift(-1)
        df["target"]        = (df["future_return"] > 0).astype(int)

        return df.dropna()


class SentimentProcessor:

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            return pd.DataFrame(columns=["date", "sentiment_score"])

        df             = news_df.copy()
        df["compound"] = df["headline"].apply(
            lambda h: self.analyzer.polarity_scores(h)["compound"]
        )

        daily = (
            df.groupby("date")["compound"]
            .mean()
            .reset_index()
            .rename(columns={"compound": "sentiment_score"})
        )

        daily["date"] = pd.to_datetime(daily["date"])
        return daily

    def merge_with_prices(
        self,
        price_df:     pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:

        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

        merged = price_df.merge(
            sentiment_df,
            left_index=True,
            right_on="date",
            how="left",
        ).set_index("date")

        merged["sentiment_score"] = (
            merged["sentiment_score"]
            .ffill()
            .fillna(0)
        )

        return merged

    def process(
        self,
        price_df: pd.DataFrame,
        news_df:  pd.DataFrame,
    ) -> pd.DataFrame:

        sentiment_df = self.score_headlines(news_df)
        return self.merge_with_prices(price_df, sentiment_df)