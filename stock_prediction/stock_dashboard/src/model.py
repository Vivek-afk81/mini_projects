# =============================================================================
# model.py
# Responsible for LightGBM direction prediction.
# Walk-forward cross validation, backtesting, save/load pickle.
# Expects fully featured DataFrames from processor.py.
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from settings import (
    FEATURES,
    LGBM_PARAMS,
    LGBM_MODEL_PATH,
    N_SPLITS,
    PREDICTION_THRESHOLD,
    RISK_PARAMS,
)


class DirectionModel:

    def __init__(self):
        self.model      = None
        self.cv_scores  = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DirectionModel":
        X = df[FEATURES]
        y = df["target"]

        self.model = lgb.LGBMClassifier(**LGBM_PARAMS)
        self.model.fit(X, y)

        return self

    # -------------------------------------------------------------------------
    # Walk-Forward Cross Validation
    # -------------------------------------------------------------------------

    def walk_forward_validate(self, df: pd.DataFrame) -> dict:
        X      = df[FEATURES]
        y      = df["target"]
        tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
        folds  = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            m = lgb.LGBMClassifier(**LGBM_PARAMS)
            m.fit(X_train, y_train)

            probs = m.predict_proba(X_test)[:, 1]
            preds = (probs >= PREDICTION_THRESHOLD).astype(int)

            folds.append({
                "fold":     fold + 1,
                "accuracy": accuracy_score(y_test, preds),
                "roc_auc":  roc_auc_score(y_test, probs),
                "n_test":   len(y_test),
            })

        self.cv_scores = pd.DataFrame(folds)

        return {
            "mean_accuracy": self.cv_scores["accuracy"].mean(),
            "std_accuracy":  self.cv_scores["accuracy"].std(),
            "mean_auc":      self.cv_scores["roc_auc"].mean(),
            "cv_detail":     self.cv_scores,
        }

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(
        self,
        df:        pd.DataFrame,
        threshold: float = PREDICTION_THRESHOLD,
    ) -> pd.DataFrame:

        probs = self.model.predict_proba(df[FEATURES])[:, 1]
        preds = (probs >= threshold).astype(int)

        result           = df.copy()
        result["prob"]   = probs
        result["signal"] = preds

        return result

    # -------------------------------------------------------------------------
    # Backtesting
    # -------------------------------------------------------------------------

    def backtest(
        self,
        df:   pd.DataFrame,
        cost: float = RISK_PARAMS["transaction_cost"],
    ) -> pd.DataFrame:

        bt = df.copy()

        bt["position"]        = bt["signal"]
        bt["strategy_return"] = bt["position"] * bt["future_return"]
        bt["trade"]           = bt["position"].diff().abs().fillna(0)
        bt["net_return"]      = bt["strategy_return"] - bt["trade"] * cost
        bt["equity_curve"]    = (1 + bt["net_return"]).cumprod()
        bt["buy_hold"]        = (1 + bt["return"]).cumprod()
        bt["drawdown"]        = bt["equity_curve"] / bt["equity_curve"].cummax() - 1

        return bt

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def compute_metrics(self, bt: pd.DataFrame) -> dict:
        y_true = bt["target"]
        y_pred = bt["signal"]
        probs  = bt["prob"]

        return {
            "total_return": bt["equity_curve"].iloc[-1] - 1,
            "buy_hold":     bt["buy_hold"].iloc[-1] - 1,
            "sharpe":       bt["net_return"].mean() / bt["net_return"].std() * np.sqrt(252),
            "max_drawdown": bt["drawdown"].min(),
            "win_rate":     (bt["net_return"] > 0).mean(),
            "accuracy":     accuracy_score(y_true, y_pred),
            "roc_auc":      roc_auc_score(y_true, probs),
            "n_trades":     int(bt["trade"].sum()),
        }

    # -------------------------------------------------------------------------
    # Feature Importance
    # -------------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature":    FEATURES,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path=LGBM_MODEL_PATH):
        joblib.dump(self.model, path)

    def load(self, path=LGBM_MODEL_PATH) -> "DirectionModel":
        self.model = joblib.load(path)
        return self