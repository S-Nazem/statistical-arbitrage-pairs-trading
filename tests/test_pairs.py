import pandas as pd

from stat_arb_research.pairs import generate_rule_signals, run_rule_based_pairs_strategy


def test_rule_signals_enter_and_exit_positions():
    zscore = pd.Series([0.0, 2.5, 1.4, 0.2, -2.3, -1.3, 0.1])

    signals = generate_rule_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

    assert signals.tolist() == [0, -1, -1, 0, 1, 1, 0]


def test_rule_based_pairs_strategy_returns_aligned_series():
    returns = pd.DataFrame(
        {
            "stock_a": [0.01, 0.02, -0.01, 0.01, 0.02, -0.01, 0.01] * 40,
            "stock_b": [0.011, 0.018, -0.012, 0.012, 0.019, -0.011, 0.009] * 40,
        },
        index=pd.date_range("2020-01-01", periods=280, freq="B"),
    )

    result = run_rule_based_pairs_strategy(
        returns,
        "stock_a",
        "stock_b",
        short_window=5,
        long_window=20,
        entry_threshold=1.0,
        exit_threshold=0.2,
    )

    assert result.strategy_returns.index.equals(returns.index)
    assert result.signals.index.equals(returns.index)
