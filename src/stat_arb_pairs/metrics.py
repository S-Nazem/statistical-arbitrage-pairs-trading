from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def cumulative_returns(returns):
    """Convert simple daily returns to cumulative returns."""
    return (1 + returns).cumprod() - 1


def annualized_sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Calculate annualized Sharpe ratio, assuming zero risk-free rate."""
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return 0.0

    volatility = clean.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return 0.0

    return float(clean.mean() / volatility * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a simple-return series."""
    clean = pd.Series(returns).fillna(0)
    equity = (1 + clean).cumprod()
    running_high = equity.cummax()
    drawdown = equity / running_high - 1
    return float(drawdown.min())


def rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    return returns.rolling(window).apply(
        lambda sample: annualized_sharpe(pd.Series(sample), periods_per_year),
        raw=False,
    )
