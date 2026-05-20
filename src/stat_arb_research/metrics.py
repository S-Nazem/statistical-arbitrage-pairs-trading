from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert simple returns to cumulative returns."""

    return (1.0 + returns).cumprod() - 1.0


def z_score(values: pd.Series) -> pd.Series:
    std = values.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=values.index, name=values.name)
    return (values - values.mean()) / std


def annualized_sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    clean = returns.dropna()
    volatility = clean.std()
    if clean.empty or volatility == 0 or np.isnan(volatility):
        return float("nan")
    return float((clean.mean() / volatility) * np.sqrt(periods_per_year))


def rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    return (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(
        periods_per_year
    )


def drawdown(returns: pd.Series) -> pd.Series:
    equity = returns.fillna(0.0).cumsum()
    return equity - equity.cummax()
