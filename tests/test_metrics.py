import pandas as pd

from stat_arb_pairs.metrics import annualized_sharpe, cumulative_returns, max_drawdown


def test_cumulative_returns_compounds_simple_returns():
    returns = pd.Series([0.1, -0.05, 0.02])

    result = cumulative_returns(returns)

    assert round(result.iloc[-1], 6) == round((1.1 * 0.95 * 1.02) - 1, 6)


def test_annualized_sharpe_handles_zero_volatility():
    returns = pd.Series([0.0, 0.0, 0.0])

    assert annualized_sharpe(returns) == 0.0


def test_max_drawdown_uses_equity_curve_peak_to_trough():
    returns = pd.Series([0.1, -0.2, 0.05])

    assert round(max_drawdown(returns), 6) == -0.2

