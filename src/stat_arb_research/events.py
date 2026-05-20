from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from stat_arb_research.metrics import annualized_sharpe, cumulative_returns, rolling_sharpe


@dataclass
class EventStudyResult:
    stock: str
    event_windows: pd.DataFrame
    mean_path: pd.Series
    fitted_path: pd.Series
    polynomial_coefficients: np.ndarray
    rmse: float


def event_windows(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock: str = "stock_1",
    window: int = 5,
) -> pd.DataFrame:
    cumulative = cumulative_returns(returns)
    rows = []
    index = list(range(-window, window + 1))

    for event_date in events.index[events[stock] == 1]:
        start = event_date - pd.offsets.BDay(window)
        end = event_date + pd.offsets.BDay(window)
        observed = cumulative[stock].loc[start:end]
        if len(observed) == (2 * window + 1):
            rows.append(observed.values - observed.iloc[window])

    return pd.DataFrame(rows, columns=index)


def fit_event_study(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock: str = "stock_1",
    window: int = 5,
) -> EventStudyResult:
    windows = event_windows(returns, events, stock=stock, window=window)
    mean_path = windows.mean(axis=0)

    x = np.array(mean_path.index, dtype=float).reshape(-1, 1)
    y = mean_path.to_numpy()
    transformer = PolynomialFeatures(degree=2)
    x_poly = transformer.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    fitted = pd.Series(model.predict(x_poly), index=mean_path.index, name="fitted_path")
    rmse = float(np.sqrt(mean_squared_error(y, fitted)))

    return EventStudyResult(
        stock=stock,
        event_windows=windows,
        mean_path=mean_path.rename("mean_path"),
        fitted_path=fitted,
        polynomial_coefficients=model.coef_,
        rmse=rmse,
    )


def event_rule_signals(
    events: pd.DataFrame,
    stock: str,
    coefficients: np.ndarray,
    window: int = 5,
) -> pd.Series:
    _, linear, quadratic = coefficients
    if abs(quadratic) < 1e-10:
        return pd.Series(0, index=events.index, name="signal")

    turning_point = int(round(-(linear / (2 * quadratic))))
    signals = pd.Series(0, index=events.index, name="signal")

    for event_ix in np.flatnonzero(events[stock].to_numpy() == 1):
        center = event_ix + turning_point
        left = center - window
        right = center + window
        if left < 0 or right >= len(events):
            continue

        if quadratic > 0:
            signals.iloc[center] = 1
            signals.iloc[left] = -1
            signals.iloc[right] = -1
        else:
            signals.iloc[center] = -1
            signals.iloc[left] = 1
            signals.iloc[right] = 1

    return signals


def run_event_strategy(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock: str = "stock_1",
    window: int = 5,
    rmse_threshold: float = 0.01,
) -> dict[str, object]:
    study = fit_event_study(returns, events, stock=stock, window=window)
    cumulative = cumulative_returns(returns)[stock]
    signals = event_rule_signals(events, stock, study.polynomial_coefficients, window=window)

    if study.rmse > rmse_threshold:
        signals = signals * 0

    strategy_returns = signals.shift(1).fillna(0.0) * cumulative.diff()
    return {
        "study": study,
        "signals": signals,
        "strategy_returns": strategy_returns.rename("strategy_return"),
        "rolling_sharpe": rolling_sharpe(strategy_returns),
        "sharpe": annualized_sharpe(strategy_returns),
    }


def run_ml_event_strategy(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock: str = "stock_1",
    window: int = 5,
    training_window: int = 10,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> dict[str, object]:
    study = fit_event_study(returns, events, stock=stock, window=window)
    labels = event_rule_signals(events, stock, study.polynomial_coefficients, window=window)
    cumulative = cumulative_returns(returns)[stock]
    features = pd.DataFrame(
        {
            "volatility": cumulative.rolling(training_window, closed="left").std(),
            "short_ma": cumulative.rolling(training_window, closed="left").mean(),
            "long_ma": cumulative.rolling(5 * training_window, closed="left").mean(),
            "return_lag_1": cumulative.diff().shift(1),
        }
    ).dropna()
    labels = labels.reindex(features.index)

    split = int(len(features) * train_fraction)
    x_train, x_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(features), index=features.index, name="signal")
    strategy_returns = predictions.shift(1).fillna(0.0) * cumulative.diff().reindex(features.index)

    return {
        "study": study,
        "signals": predictions,
        "strategy_returns": strategy_returns.rename("strategy_return"),
        "rolling_sharpe": rolling_sharpe(strategy_returns),
        "sharpe": annualized_sharpe(strategy_returns),
        "train_score": float(accuracy_score(y_train, model.predict(x_train))),
        "test_score": (
            float(accuracy_score(y_test, model.predict(x_test))) if len(x_test) else float("nan")
        ),
    }
