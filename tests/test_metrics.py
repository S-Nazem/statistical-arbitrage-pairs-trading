import numpy as np
import pandas as pd

from stat_arb_research.metrics import annualized_sharpe, cumulative_returns, z_score


def test_cumulative_returns_compounds_simple_returns():
    returns = pd.Series([0.10, -0.05, 0.02])

    result = cumulative_returns(returns)

    assert np.isclose(result.iloc[-1], (1.10 * 0.95 * 1.02) - 1)


def test_annualized_sharpe_handles_flat_returns():
    returns = pd.Series([0.0, 0.0, 0.0])

    assert np.isnan(annualized_sharpe(returns))


def test_z_score_standardizes_series():
    values = pd.Series([1.0, 2.0, 3.0])

    result = z_score(values)

    assert np.isclose(result.mean(), 0.0)
    assert np.isclose(result.std(), 1.0)
