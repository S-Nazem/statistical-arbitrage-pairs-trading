import pandas as pd
import pytest

from stat_arb_pairs.pairs import PairsTradingStrategy


def test_pairs_strategy_requires_short_window_below_long_window():
    with pytest.raises(ValueError):
        PairsTradingStrategy("stock_a", "stock_b", short_window=20, long_window=10)


def test_pairs_strategy_outputs_aligned_returns_and_positions():
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    returns = pd.DataFrame(
        {
            "stock_a": [0.001] * 120,
            "stock_b": [0.0005] * 120,
        },
        index=dates,
    )

    result = PairsTradingStrategy("stock_a", "stock_b", short_window=5, long_window=20).backtest(
        returns
    )

    assert result.strategy_returns.index.equals(returns.index)
    assert result.positions.index.equals(returns.index)

