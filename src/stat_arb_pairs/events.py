from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stat_arb_pairs.metrics import annualized_sharpe, max_drawdown, rolling_sharpe


@dataclass(frozen=True)
class EventBacktestResult:
    strategy_returns: pd.Series
    positions: pd.Series
    average_event_path: pd.Series
    sharpe: float
    max_drawdown: float
    rolling_sharpe: pd.Series


class EventStudyStrategy:
    """Trade in the direction of the estimated post-event drift."""

    def __init__(self, stock: str, window: int = 5, hold_days: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be positive.")
        if hold_days < 1:
            raise ValueError("hold_days must be positive.")

        self.stock = stock
        self.window = window
        self.hold_days = hold_days

    def average_event_path(self, returns: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
        self._validate_inputs(returns, events)
        cumulative = (1 + returns[self.stock]).cumprod() - 1
        event_dates = events.index[events[self.stock] == 1]
        windows = []

        for event_date in event_dates:
            start = event_date - pd.offsets.BDay(self.window)
            end = event_date + pd.offsets.BDay(self.window)
            sample = cumulative.loc[start:end]
            expected_length = self.window * 2 + 1
            if len(sample) == expected_length:
                windows.append((sample - sample.iloc[self.window]).to_numpy())

        if not windows:
            raise ValueError(f"No complete event windows found for {self.stock}.")

        index = pd.RangeIndex(-self.window, self.window + 1, name="days_from_event")
        return pd.Series(np.mean(windows, axis=0), index=index, name=self.stock)

    def backtest(self, returns: pd.DataFrame, events: pd.DataFrame) -> EventBacktestResult:
        self._validate_inputs(returns, events)
        average_path = self.average_event_path(returns, events)
        post_event_drift = average_path.loc[1 : self.hold_days].sum()
        direction = 1 if post_event_drift >= 0 else -1

        positions = pd.Series(0, index=returns.index, name="position")
        event_dates = events.index[events[self.stock] == 1]
        for event_date in event_dates:
            loc = returns.index.get_indexer([event_date])[0]
            if loc == -1:
                continue
            start = loc
            stop = min(loc + self.hold_days, len(positions))
            positions.iloc[start:stop] = direction

        strategy_returns = positions.shift(1).fillna(0) * returns[self.stock]
        strategy_returns.name = f"{self.stock}_event_strategy_return"

        return EventBacktestResult(
            strategy_returns=strategy_returns,
            positions=positions,
            average_event_path=average_path,
            sharpe=annualized_sharpe(strategy_returns),
            max_drawdown=max_drawdown(strategy_returns),
            rolling_sharpe=rolling_sharpe(strategy_returns),
        )

    def _validate_inputs(self, returns: pd.DataFrame, events: pd.DataFrame) -> None:
        if self.stock not in returns.columns:
            raise KeyError(f"{self.stock} not found in returns data.")
        if self.stock not in events.columns:
            raise KeyError(f"{self.stock} not found in events data.")
        if not returns.index.equals(events.index):
            raise ValueError("Returns and events must share the same DatetimeIndex.")
