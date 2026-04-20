"""Train a gradient-boosted tree regressor for next-day close; backtest and multi-step forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import RANDOM_STATE, TEST_SIZE_FRAC
from src.features import build_feature_matrix, feature_columns


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


@dataclass
class BacktestResult:
    y_test: pd.Series
    y_pred: pd.Series
    mae: float
    rmse: float
    mape: float


@dataclass
class TrainForecastResult:
    model: CatBoostRegressor
    feature_names: list[str]
    backtest: BacktestResult
    history_close: pd.Series
    forecast_index: pd.DatetimeIndex
    forecast_close: np.ndarray


def train_and_backtest(close: pd.Series, test_size_frac: float = TEST_SIZE_FRAC) -> TrainForecastResult:
    """
    Time-ordered split: last `test_size_frac` of rows for walk-forward-style
    one-step-ahead predictions (retrain on expanding... actually we use single
    train on first portion, test on last portion for speed; for production you'd use rolling refit).
    """
    X, y = build_feature_matrix(close)
    if len(X) < 50:
        raise ValueError("Not enough rows after feature construction; try a longer history.")

    n = len(X)
    split = int(n * (1 - test_size_frac))
    split = max(split, n // 2)
    split = min(split, n - 10)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    feats = feature_columns(X_train)
    model = CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        subsample=0.9,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_train[feats], y_train)

    y_pred = model.predict(X_test[feats])
    y_test_vals = y_test.values
    mae = float(mean_absolute_error(y_test_vals, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_vals, y_pred)))
    mape = _mape(y_test_vals, y_pred)

    backtest = BacktestResult(
        y_test=y_test,
        y_pred=pd.Series(y_pred, index=y_test.index),
        mae=mae,
        rmse=rmse,
        mape=mape,
    )

    # Refit on full X,y for deployment forecast
    model_full = CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        subsample=0.9,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model_full.fit(X[feats], y)

    return TrainForecastResult(
        model=model_full,
        feature_names=feats,
        backtest=backtest,
        history_close=close,
        forecast_index=pd.DatetimeIndex([]),
        forecast_close=np.array([]),
    )


def recursive_forecast(
    close: pd.Series,
    model: CatBoostRegressor,
    feature_names: list[str],
    horizon_days: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Multi-day ahead forecast: append each predicted close and rebuild the last row's features.
    Uses simplified feature update (lags shift, rolling stats approximated from extended series).
    """
    extended = close.astype(float).copy()
    last_date = extended.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=horizon_days,
    )

    preds = []
    for i in range(horizon_days):
        X_full, _ = build_feature_matrix(extended)
        if X_full.empty:
            break
        row = X_full.iloc[[-1]][feature_names]
        next_close = float(model.predict(row)[0])
        preds.append(next_close)
        new_idx = future_dates[i]
        extended = pd.concat([extended, pd.Series([next_close], index=[new_idx])])

    return future_dates[: len(preds)], np.array(preds, dtype=float)


def attach_forecast(
    result: TrainForecastResult,
    horizon_days: int,
) -> TrainForecastResult:
    idx, fc = recursive_forecast(
        result.history_close,
        result.model,
        result.feature_names,
        horizon_days,
    )
    return TrainForecastResult(
        model=result.model,
        feature_names=result.feature_names,
        backtest=result.backtest,
        history_close=result.history_close,
        forecast_index=idx,
        forecast_close=fc,
    )