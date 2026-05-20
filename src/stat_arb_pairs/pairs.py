from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from stat_arb_pairs.metrics import annualized_sharpe, cumulative_returns, max_drawdown, rolling_sharpe


@dataclass(frozen=True)
class PairCandidate:
    stock_a: str
    stock_b: str
    correlation: float
    cointegration_pvalue: float


@dataclass(frozen=True)
class BacktestResult:
    strategy_returns: pd.Series
    positions: pd.Series
    spread: pd.Series
    z_score: pd.Series
    sharpe: float
    max_drawdown: float
    rolling_sharpe: pd.Series


def find_candidate_pairs(
    returns: pd.DataFrame,
    min_correlation: float = 0.85,
    max_cointegration_pvalue: float = 0.05,
    max_pairs: int = 10,
) -> list[PairCandidate]:
    """Rank pairs by cumulative-return correlation and cointegration p-value."""
    cumulative = cumulative_returns(returns)
    correlations = cumulative.corr()
    candidates: list[PairCandidate] = []

    for stock_a, stock_b in combinations(cumulative.columns, 2):
        corr = float(correlations.loc[stock_a, stock_b])
        if abs(corr) < min_correlation:
            continue

        _, pvalue, _ = coint(cumulative[stock_a], cumulative[stock_b])
        if pvalue <= max_cointegration_pvalue:
            candidates.append(PairCandidate(stock_a, stock_b, corr, float(pvalue)))

    return sorted(candidates, key=lambda pair: (-abs(pair.correlation), pair.cointegration_pvalue))[
        :max_pairs
    ]


class PairsTradingStrategy:
    """Moving-average z-score pairs strategy on cumulative-return spreads."""

    def __init__(
        self,
        stock_a: str,
        stock_b: str,
        short_window: int = 10,
        long_window: int = 50,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window.")
        if entry_threshold <= exit_threshold:
            raise ValueError("entry_threshold must be larger than exit_threshold.")

        self.stock_a = stock_a
        self.stock_b = stock_b
        self.short_window = short_window
        self.long_window = long_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def backtest(self, returns: pd.DataFrame) -> BacktestResult:
        missing = {self.stock_a, self.stock_b}.difference(returns.columns)
        if missing:
            raise KeyError(f"Missing stocks in returns data: {sorted(missing)}")

        cumulative = cumulative_returns(returns[[self.stock_a, self.stock_b]])
        spread = cumulative[self.stock_a] - cumulative[self.stock_b]

        short_ma = spread.rolling(self.short_window).mean()
        long_ma = spread.rolling(self.long_window).mean()
        ma_spread = short_ma - long_ma
        z_score = ma_spread / ma_spread.rolling(self.long_window).std()

        positions = self._generate_positions(z_score)
        spread_returns = returns[self.stock_a] - returns[self.stock_b]
        strategy_returns = positions.shift(1).fillna(0) * spread_returns
        strategy_returns.name = f"{self.stock_a}_{self.stock_b}_strategy_return"

        return BacktestResult(
            strategy_returns=strategy_returns,
            positions=positions,
            spread=spread,
            z_score=z_score,
            sharpe=annualized_sharpe(strategy_returns),
            max_drawdown=max_drawdown(strategy_returns),
            rolling_sharpe=rolling_sharpe(strategy_returns),
        )

    def _generate_positions(self, z_score: pd.Series) -> pd.Series:
        position = 0
        positions = []

        for value in z_score:
            if np.isnan(value):
                positions.append(0)
                continue

            if value > self.entry_threshold:
                position = -1
            elif value < -self.entry_threshold:
                position = 1
            elif abs(value) < self.exit_threshold:
                position = 0

            positions.append(position)

        return pd.Series(positions, index=z_score.index, name="position")

