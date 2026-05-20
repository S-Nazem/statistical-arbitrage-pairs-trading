import pandas as pd

from stat_arb_pairs.events import EventStudyStrategy


def test_event_strategy_outputs_aligned_returns_and_positions():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    returns = pd.DataFrame({"stock_a": [0.001] * 30}, index=dates)
    events = pd.DataFrame({"stock_a": [0] * 30}, index=dates)
    events.iloc[[6, 16], 0] = 1

    result = EventStudyStrategy("stock_a", window=3, hold_days=2).backtest(returns, events)

    assert result.strategy_returns.index.equals(returns.index)
    assert result.positions.index.equals(returns.index)
    assert result.average_event_path.index.tolist() == [-3, -2, -1, 0, 1, 2, 3]

