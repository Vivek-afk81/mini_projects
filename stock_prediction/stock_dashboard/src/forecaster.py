# =============================================================================
# forecaster.py
# Responsible for Prophet price trend forecasting.
# One Prophet model trained and saved per ticker.
# Expects raw price DataFrames from extractor.py.
# =============================================================================

import pandas as pd
import joblib
from prophet import Prophet

from settings import (
    PROPHET_PARAMS,
    PROPHET_FORECAST_DAYS,
    prophet_model_path,
)


class PriceForecaster:

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model  = None

    # -------------------------------------------------------------------------
    # Data Preparation
    # Prophet expects exactly two columns: ds (date) and y (value)
    # -------------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df.index).tz_localize(None),
            "y":  df["Close"].values,
        })
        return prophet_df

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PriceForecaster":
        self.model = Prophet(**PROPHET_PARAMS)

        prophet_df = self._prepare(df)
        self.model.fit(prophet_df)

        return self

    # -------------------------------------------------------------------------
    # Forecasting
    # -------------------------------------------------------------------------

    def forecast(
        self,
        days: int = PROPHET_FORECAST_DAYS,
    ) -> pd.DataFrame:

        future   = self.model.make_future_dataframe(periods=days)
        forecast = self.model.predict(future)

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.rename(columns={
            "ds":          "date",
            "yhat":        "predicted_price",
            "yhat_lower":  "lower_bound",
            "yhat_upper":  "upper_bound",
        })

        result["date"] = pd.to_datetime(result["date"])
        return result.set_index("date")

    # -------------------------------------------------------------------------
    # Trend Signal
    # Returns a simple scalar: positive = uptrend, negative = downtrend
    # Used by optimiser.py to adjust allocation weights
    # -------------------------------------------------------------------------

    def trend_signal(self, days: int = PROPHET_FORECAST_DAYS) -> float:
        fc           = self.forecast(days=days)
        future_only  = fc.iloc[-days:]
        start_price  = future_only["predicted_price"].iloc[0]
        end_price    = future_only["predicted_price"].iloc[-1]

        return (end_price - start_price) / start_price

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self):
        path = prophet_model_path(self.ticker)
        joblib.dump(self.model, path)

    def load(self) -> "PriceForecaster":
        path        = prophet_model_path(self.ticker)
        self.model  = joblib.load(path)
        return self