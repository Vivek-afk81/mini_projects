# =============================================================================
# optimiser.py
# Responsible for portfolio optimisation using PyPortfolioOpt.
# Takes price data, LightGBM signals and Prophet trend signals as inputs.
# Returns final portfolio weights and performance metrics.
# =============================================================================

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

from settings import RISK_PARAMS, TICKERS


class PortfolioOptimiser:

    def __init__(self):
        self.rf       = RISK_PARAMS["risk_free_rate"]
        self.max_w    = RISK_PARAMS["max_weight"]
        self.min_w    = RISK_PARAMS["min_weight"]
        self.weights  = {}
        self.metrics  = {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_price_matrix(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        closes = {
            ticker: df["Close"]
            for ticker, df in price_data.items()
        }

        prices = pd.DataFrame(closes).dropna()
        prices.index = pd.to_datetime(prices.index).tz_localize(None)

        return prices

    def _adjust_expected_returns(
        self,
        mu:             pd.Series,
        trend_signals:  dict[str, float],
    ) -> pd.Series:

        mu = mu.copy()

        for ticker, signal in trend_signals.items():
            if ticker in mu.index:
                # Nudge expected return up or down based on Prophet trend
                mu[ticker] = mu[ticker] * (1 + signal)

        return mu

    # -------------------------------------------------------------------------
    # Core Optimisation
    # -------------------------------------------------------------------------

    def optimise(
        self,
        price_data:     dict[str, pd.DataFrame],
        trend_signals:  dict[str, float] = None,
    ) -> dict:

        prices = self._build_price_matrix(price_data)

        mu  = expected_returns.mean_historical_return(prices)
        cov = risk_models.sample_cov(prices)

        if trend_signals:
            mu = self._adjust_expected_returns(mu, trend_signals)

        ef = EfficientFrontier(mu, cov)
        ef.add_constraint(lambda w: w <= self.max_w)
        ef.add_constraint(lambda w: w >= self.min_w)
        ef.max_sharpe(risk_free_rate=self.rf)

        self.weights = dict(ef.clean_weights())

        perf = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=self.rf,
        )

        self.metrics = {
            "expected_annual_return": perf[0],
            "annual_volatility":      perf[1],
            "sharpe_ratio":           perf[2],
        }

        return {
            "weights": self.weights,
            "metrics": self.metrics,
        }

    # -------------------------------------------------------------------------
    # Signal Filtering
    # Only allocate to tickers where LightGBM signal is bullish
    # -------------------------------------------------------------------------

    def filter_by_signals(
        self,
        price_data:    dict[str, pd.DataFrame],
        lgbm_signals:  dict[str, int],
        trend_signals: dict[str, float] = None,
    ) -> dict:

        bullish_tickers = [
            ticker for ticker, signal in lgbm_signals.items()
            if signal == 1
        ]

        if not bullish_tickers:
            return {
                "weights": {t: 0.0 for t in price_data.keys()},
                "metrics": {
                    "expected_annual_return": 0.0,
                    "annual_volatility":      0.0,
                    "sharpe_ratio":           0.0,
                },
                "note": "No bullish signals — all weights set to zero.",
            }

        filtered_prices = {
            t: price_data[t]
            for t in bullish_tickers
            if t in price_data
        }

        filtered_trends = None
        if trend_signals:
            filtered_trends = {
                t: trend_signals[t]
                for t in bullish_tickers
                if t in trend_signals
            }

        result = self.optimise(filtered_prices, filtered_trends)

        # Fill non-bullish tickers with zero weight for completeness
        all_weights = {t: 0.0 for t in price_data.keys()}
        all_weights.update(result["weights"])

        return {
            "weights": all_weights,
            "metrics": result["metrics"],
        }

    # -------------------------------------------------------------------------
    # Weights as DataFrame for display
    # -------------------------------------------------------------------------

    def weights_to_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame
            .from_dict(self.weights, orient="index", columns=["weight"])
            .sort_values("weight", ascending=False)
            .reset_index()
            .rename(columns={"index": "ticker"})
        )