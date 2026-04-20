"""Lag and rolling features for price forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import LAG_DAYS, ROLL_WINDOWS


def build_feature_matrix(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (features) and y (next-day close). Rows with NaN from lags/rolls are dropped.
    """
    s = close.astype(float)
    out = pd.DataFrame(index=s.index)
    out["close"] = s

    for lag in LAG_DAYS:
        out[f"lag_{lag}"] = s.shift(lag)

    out["ret_1"] = s.pct_change(1)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = s.rolling(w).mean()
        out[f"roll_std_{w}"] = s.rolling(w).std()

    y = s.shift(-1)
    mask = out.notna().all(axis=1) & y.notna()
    X = out.loc[mask].copy()
    y_aligned = y.loc[mask].copy()
    return X, y_aligned


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != "close"]
