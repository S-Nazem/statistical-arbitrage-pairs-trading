import numpy as np
import pandas as pd

from stat_arb_research.events import event_rule_signals, event_windows, fit_event_study


def _sample_returns_and_events():
    index = pd.date_range("2021-01-01", periods=40, freq="B")
    returns = pd.DataFrame({"stock_1": np.linspace(-0.01, 0.01, len(index))}, index=index)
    events = pd.DataFrame({"stock_1": 0}, index=index)
    events.iloc[[10, 20, 30], 0] = 1
    return returns, events


def test_event_windows_are_centered_on_event_day():
    returns, events = _sample_returns_and_events()

    windows = event_windows(returns, events, "stock_1", window=3)

    assert list(windows.columns) == [-3, -2, -1, 0, 1, 2, 3]
    assert (windows[0] == 0).all()


def test_fit_event_study_returns_polynomial_path():
    returns, events = _sample_returns_and_events()

    result = fit_event_study(returns, events, "stock_1", window=3)

    assert len(result.mean_path) == 7
    assert len(result.fitted_path) == 7
    assert result.rmse >= 0


def test_event_rule_signals_generates_trades_around_turning_point():
    _, events = _sample_returns_and_events()
    coefficients = np.array([0.0, 0.0, 1.0])

    signals = event_rule_signals(events, "stock_1", coefficients, window=2)

    assert set(signals.unique()) == {-1, 0, 1}
