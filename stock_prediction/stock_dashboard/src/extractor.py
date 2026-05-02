# extractor.py
# Responsible for fetching raw data only.
# No feature engineering. No sentiment scoring. No model logic.


import feedparser
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from settings import DATE_CONFIG, NEWS_FEEDS, NEWS_LOOKBACK_DAYS, TICKERS


class MarketExtractor:

    def fetch(
        self,
        ticker: str,
        start: str = DATE_CONFIG["start"],
        end: str   = DATE_CONFIG["end"],
    ) -> pd.DataFrame:

        df = yf.download(ticker, start=start, end=end, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["return"] = df["Close"].pct_change()

        return df.dropna()

    def fetch_multiple(
        self,
        tickers: list = TICKERS,
        start: str    = DATE_CONFIG["start"],
        end: str      = DATE_CONFIG["end"],
    ) -> dict[str, pd.DataFrame]:

        return {ticker: self.fetch(ticker, start, end) for ticker in tickers}


class NewsExtractor:

    def fetch_headlines(
        self,
        days_back: int = NEWS_LOOKBACK_DAYS,
    ) -> pd.DataFrame:

        cutoff  = datetime.now() - timedelta(days=days_back)
        records = []

        for url in NEWS_FEEDS:
            feed = feedparser.parse(url)

            for entry in feed.entries:
                try:
                    published = datetime(*entry.published_parsed[:6])

                    if published >= cutoff:
                        records.append({
                            "date":     published.date(),
                            "headline": entry.title,
                            "source":   feed.feed.get("title", url),
                        })

                except Exception:
                    continue

        if not records:
            return pd.DataFrame(columns=["date", "headline", "source"])

        df           = pd.DataFrame(records)
        df["date"]   = pd.to_datetime(df["date"])

        return df.sort_values("date").reset_index(drop=True)