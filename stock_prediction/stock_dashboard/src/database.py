# =============================================================================
# database.py
# Responsible for all SQLite operations.
# No model logic. No data fetching. No calculations.
# Every module that needs persistence goes through here.
# =============================================================================

import sqlite3
import datetime
import pandas as pd
from pathlib import Path

from settings import DB_PATH, ensure_dirs


class Database:

    def __init__(self, path: Path = DB_PATH):
        ensure_dirs()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._init_tables()

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                signal      INTEGER NOT NULL,
                prob        REAL    NOT NULL,
                actual      INTEGER
            );

            CREATE TABLE IF NOT EXISTS backtest_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker              TEXT    NOT NULL,
                run_date            TEXT    NOT NULL,
                total_return        REAL,
                buy_hold            REAL,
                sharpe              REAL,
                max_drawdown        REAL,
                win_rate            REAL,
                accuracy            REAL,
                roc_auc             REAL,
                n_trades            INTEGER
            );

            CREATE TABLE IF NOT EXISTS portfolio_weights (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date                TEXT    NOT NULL,
                ticker                  TEXT    NOT NULL,
                weight                  REAL    NOT NULL,
                expected_annual_return  REAL,
                annual_volatility       REAL,
                sharpe_ratio            REAL
            );

            CREATE TABLE IF NOT EXISTS cv_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                run_date    TEXT    NOT NULL,
                fold        INTEGER NOT NULL,
                accuracy    REAL,
                roc_auc     REAL,
                n_test      INTEGER
            );
        """)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    def save_predictions(
        self,
        ticker: str,
        df:     pd.DataFrame,
    ):
        rows = df[["signal", "prob", "target"]].copy()
        rows["ticker"] = ticker
        rows["date"]   = rows.index.astype(str)
        rows           = rows.rename(columns={"target": "actual"})

        rows.to_sql(
            "predictions",
            self.conn,
            if_exists="append",
            index=False,
        )

    def get_predictions(self, ticker: str) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM predictions WHERE ticker = ? ORDER BY date DESC",
            self.conn,
            params=(ticker,),
        )

    # -------------------------------------------------------------------------
    # Backtest Results
    # -------------------------------------------------------------------------

    def save_backtest(self, ticker: str, metrics: dict):
        row = pd.DataFrame([{
            "ticker":       ticker,
            "run_date":     str(datetime.date.today()),
            **metrics,
        }])

        row.to_sql(
            "backtest_results",
            self.conn,
            if_exists="append",
            index=False,
        )

    def get_backtest_history(self, ticker: str) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM backtest_results WHERE ticker = ? ORDER BY run_date DESC",
            self.conn,
            params=(ticker,),
        )

    # -------------------------------------------------------------------------
    # Portfolio Weights
    # -------------------------------------------------------------------------

    def save_portfolio(self, weights: dict, metrics: dict):
        run_date = str(datetime.date.today())

        rows = pd.DataFrame([
            {
                "run_date":                 run_date,
                "ticker":                   ticker,
                "weight":                   weight,
                "expected_annual_return":   metrics.get("expected_annual_return"),
                "annual_volatility":        metrics.get("annual_volatility"),
                "sharpe_ratio":             metrics.get("sharpe_ratio"),
            }
            for ticker, weight in weights.items()
        ])

        rows.to_sql(
            "portfolio_weights",
            self.conn,
            if_exists="append",
            index=False,
        )

    def get_latest_portfolio(self) -> pd.DataFrame:
        return pd.read_sql(
            """
            SELECT * FROM portfolio_weights
            WHERE run_date = (SELECT MAX(run_date) FROM portfolio_weights)
            ORDER BY weight DESC
            """,
            self.conn,
        )

    def get_portfolio_history(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM portfolio_weights ORDER BY run_date DESC",
            self.conn,
        )

    # -------------------------------------------------------------------------
    # Cross Validation Results
    # -------------------------------------------------------------------------

    def save_cv_results(self, ticker: str, cv_df: pd.DataFrame):
        rows             = cv_df.copy()
        rows["ticker"]   = ticker
        rows["run_date"] = str(datetime.date.today())

        rows.to_sql(
            "cv_results",
            self.conn,
            if_exists="append",
            index=False,
        )

    def get_cv_history(self, ticker: str) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM cv_results WHERE ticker = ? ORDER BY run_date DESC",
            self.conn,
            params=(ticker,),
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def close(self):
        self.conn.close()