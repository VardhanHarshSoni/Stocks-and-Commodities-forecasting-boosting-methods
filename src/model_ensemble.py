"""Ensemble model combining GradientBoosting, LightGBM, CatBoost, and XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import build_feature_matrix


class EnsembleModel:
    """
    Wrapper class that makes an ensemble of 4 models behave like a single model.
    Aggregates predictions and feature importances from all models using weights.
    """
    
    def __init__(self, models: dict[str, Any], weights: dict[str, float], feature_names: list[str]):
        """
        Args:
            models: Dict of trained model objects {"gbr": ..., "lightgbm": ..., "catboost": ..., "xgb": ...}
            weights: Dict of weights for each model
            feature_names: List of feature column names
        """
        self.models = models
        self.weights = weights
        self.feature_names = feature_names
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions by weighted averaging.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Ensemble predictions as numpy array
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions["gbr"], dtype=float)
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Aggregate feature importances from all models using weights.
        
        Returns:
            Weighted average of feature importances across all models
        """
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
        
        if not importances:
            # No models have feature importances, return zeros
            return np.zeros(len(self.feature_names))
        
        # Initialize ensemble importances
        ensemble_imp = np.zeros_like(next(iter(importances.values())), dtype=float)
        
        # Weighted average of importances
        for name, imp in importances.items():
            ensemble_imp += self.weights[name] * imp
        
        return ensemble_imp

from config import TEST_SIZE_FRAC
from src.model_GBR import (
    train_and_backtest as train_gbr,
    attach_forecast as attach_gbr,
    TrainForecastResult as GBRResult,
)
from src.model_lightGBM import (
    train_and_backtest as train_lightgbm,
    attach_forecast as attach_lightgbm,
    TrainForecastResult as LGBMResult,
)
from src.model_CatBooster import (
    train_and_backtest as train_catboost,
    attach_forecast as attach_catboost,
    TrainForecastResult as CatBoostResult,
)
from src.model_XGB import (
    train_and_backtest as train_xgb,
    attach_forecast as attach_xgb,
    TrainForecastResult as XGBResult,
)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
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
    """Ensemble result with single model attribute for compatibility."""
    model: EnsembleModel               # The ensemble model (behaves like a single model)
    feature_names: list[str]
    backtest: BacktestResult
    history_close: pd.Series
    forecast_index: pd.DatetimeIndex
    forecast_close: np.ndarray
    weights: dict[str, float]          # Model names to weights for ensemble
    individual_models: dict[str, Any]  # Store individual model results for reference


def _compute_ensemble_weights(results: dict[str, Any]) -> dict[str, float]:
    """
    Compute weights inversely proportional to MAE.
    Models with lower error get higher weight.
    """
    weights = {}
    mae_values = {name: result.backtest.mae for name, result in results.items()}
    
    # Compute inverse weights
    inv_sum = sum(1.0 / max(mae, 1e-6) for mae in mae_values.values())
    for name, mae in mae_values.items():
        weights[name] = (1.0 / max(mae, 1e-6)) / inv_sum
    
    return weights


def train_and_backtest(close: pd.Series, test_size_frac: float = TEST_SIZE_FRAC) -> TrainForecastResult:
    """
    Train all 4 models and combine via weighted averaging (inverse MAE weighting).
    
    Args:
        close: pd.Series of historical close prices
        test_size_frac: fraction of data to hold out for backtesting
        
    Returns:
        TrainForecastResult with ensemble model and backtest metrics
    """
    print("Training ensemble: GBR, LightGBM, CatBoost, XGBoost...")
    
    X, y = build_feature_matrix(close)
    
    # Train individual models in parallel (conceptually)
    gbr_result = train_gbr(close, test_size_frac)
    lightgbm_result = train_lightgbm(close, test_size_frac)
    catboost_result = train_catboost(close, test_size_frac)
    xgb_result = train_xgb(close, test_size_frac)
    
    individual_results = {
        "gbr": gbr_result,
        "lightgbm": lightgbm_result,
        "catboost": catboost_result,
        "xgb": xgb_result,
    }
    
    # Compute ensemble weights
    weights = _compute_ensemble_weights(individual_results)
    print(f"Ensemble weights: {weights}")
    
    # Compute ensemble backtest predictions via weighted average
    y_test = gbr_result.backtest.y_test
    ensemble_pred = (
        weights["gbr"] * gbr_result.backtest.y_pred.values +
        weights["lightgbm"] * lightgbm_result.backtest.y_pred.values +
        weights["catboost"] * catboost_result.backtest.y_pred.values +
        weights["xgb"] * xgb_result.backtest.y_pred.values
    )
    
    # Compute ensemble backtest metrics
    y_test_vals = y_test.values
    mae = float(mean_absolute_error(y_test_vals, ensemble_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_vals, ensemble_pred)))
    mape = _mape(y_test_vals, ensemble_pred)
    
    backtest = BacktestResult(
        y_test=y_test,
        y_pred=pd.Series(ensemble_pred, index=y_test.index),
        mae=mae,
        rmse=rmse,
        mape=mape,
    )
    
    print(f"Ensemble Backtest - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Use feature names from first model (all should be same)
    feature_names = gbr_result.feature_names
    feats = feature_names
    
    # Extract the actual model objects from TrainForecastResult
    model_objects = {
        "gbr": gbr_result.model,
        "lightgbm": lightgbm_result.model,
        "catboost": catboost_result.model,
        "xgb": xgb_result.model,
    }
    
    # Create ensemble model
    ensemble_model = EnsembleModel(model_objects, weights, feature_names)
    
    # Predict on full data for visualization
    full_pred = ensemble_model.predict(X[feats])
    y_pred_full = pd.Series(full_pred, index=y.index)
    
    # Update backtest with full predictions for visualization
    backtest.y_pred = y_pred_full
    
    return TrainForecastResult(
        model=ensemble_model,
        feature_names=feature_names,
        backtest=backtest,
        history_close=close,
        forecast_index=pd.DatetimeIndex([]),
        forecast_close=np.array([]),
        weights=weights,
        individual_models=individual_results,
    )


def attach_forecast(
    result: TrainForecastResult,
    horizon_days: int,
) -> TrainForecastResult:
    """
    Generate ensemble forecast by combining individual model forecasts.
    
    Args:
        result: TrainForecastResult from train_and_backtest
        horizon_days: number of days to forecast ahead
        
    Returns:
        Updated TrainForecastResult with forecast_close and forecast_index
    """
    print(f"Generating {horizon_days}-day ensemble forecast...")
    
    # Attach forecasts for each individual model
    gbr_forecast = attach_gbr(result.individual_models["gbr"], horizon_days)
    lightgbm_forecast = attach_lightgbm(result.individual_models["lightgbm"], horizon_days)
    catboost_forecast = attach_catboost(result.individual_models["catboost"], horizon_days)
    xgb_forecast = attach_xgb(result.individual_models["xgb"], horizon_days)
    
    # Combine forecasts via weighted average
    ensemble_forecast = (
        result.weights["gbr"] * gbr_forecast.forecast_close +
        result.weights["lightgbm"] * lightgbm_forecast.forecast_close +
        result.weights["catboost"] * catboost_forecast.forecast_close +
        result.weights["xgb"] * xgb_forecast.forecast_close
    )
    
    # Use forecast index from first model (all should align)
    forecast_index = gbr_forecast.forecast_index
    
    return TrainForecastResult(
        model=result.model,
        feature_names=result.feature_names,
        backtest=result.backtest,
        history_close=result.history_close,
        forecast_index=forecast_index,
        forecast_close=ensemble_forecast,
        weights=result.weights,
        individual_models=result.individual_models,
    )
