from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statsmodels.tsa.stattools import adfuller, coint

from stat_arb_research.metrics import annualized_sharpe, cumulative_returns, rolling_sharpe


@dataclass(frozen=True)
class PairDiagnostics:
    stock_a: str
    stock_b: str
    correlation: float
    cointegration_pvalue: float
    spread_adf_pvalue: float
    ratio_adf_pvalue: float

    @property
    def is_eligible(self) -> bool:
        return self.correlation >= 0.8 and self.spread_adf_pvalue < 0.05


@dataclass
class PairStrategyResult:
    stock_a: str
    stock_b: str
    spread: pd.Series
    zscore: pd.Series
    signals: pd.Series
    strategy_returns: pd.Series
    rolling_sharpe: pd.Series
    diagnostics: PairDiagnostics
    train_score: float | None = None
    test_score: float | None = None

    @property
    def sharpe(self) -> float:
        return annualized_sharpe(self.strategy_returns)


def spread(cumulative: pd.DataFrame, stock_a: str, stock_b: str) -> pd.Series:
    return cumulative[stock_a] - cumulative[stock_b]


def moving_average_zscore(
    pair_spread: pd.Series,
    short_window: int,
    long_window: int,
    *,
    closed: str | None = None,
) -> pd.Series:
    short_ma = pair_spread.rolling(short_window, closed=closed).mean()
    long_ma = pair_spread.rolling(long_window, closed=closed).mean()
    ma_diff = short_ma - long_ma
    return ma_diff / ma_diff.rolling(long_window).std()


def pair_diagnostics(returns: pd.DataFrame, stock_a: str, stock_b: str) -> PairDiagnostics:
    cumulative = cumulative_returns(returns)
    pair_spread = spread(cumulative, stock_a, stock_b).dropna()
    ratio = (cumulative[stock_a] / cumulative[stock_b]).replace([np.inf, -np.inf], np.nan).dropna()

    return PairDiagnostics(
        stock_a=stock_a,
        stock_b=stock_b,
        correlation=float(cumulative[stock_a].corr(cumulative[stock_b])),
        cointegration_pvalue=float(coint(cumulative[stock_a], cumulative[stock_b])[1]),
        spread_adf_pvalue=float(adfuller(pair_spread)[1]),
        ratio_adf_pvalue=float(adfuller(ratio)[1]) if len(ratio) > 20 else float("nan"),
    )


def find_correlated_pairs(
    returns: pd.DataFrame,
    base_stock: str = "stock_0",
    top_n: int = 5,
) -> pd.DataFrame:
    cumulative = cumulative_returns(returns)
    correlations = cumulative.corr()[base_stock].sort_values(ascending=False)
    rows = []
    for stock_b, corr_value in correlations.drop(index=base_stock).head(top_n).items():
        diagnostics = pair_diagnostics(returns, base_stock, stock_b)
        rows.append(
            {
                "stock_a": base_stock,
                "stock_b": stock_b,
                "correlation": corr_value,
                "cointegration_pvalue": diagnostics.cointegration_pvalue,
                "spread_adf_pvalue": diagnostics.spread_adf_pvalue,
                "ratio_adf_pvalue": diagnostics.ratio_adf_pvalue,
                "eligible": diagnostics.is_eligible,
            }
        )
    return pd.DataFrame(rows)


def generate_rule_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 1.0,
) -> pd.Series:
    position = 0
    signals: list[int] = []
    for z_value in zscore:
        if np.isnan(z_value):
            signals.append(position)
            continue
        if position == 0 and z_value > entry_threshold:
            position = -1
        elif position == 0 and z_value < -entry_threshold:
            position = 1
        elif position != 0 and abs(z_value) < exit_threshold:
            position = 0
        signals.append(position)
    return pd.Series(signals, index=zscore.index, name="signal")


def run_rule_based_pairs_strategy(
    returns: pd.DataFrame,
    stock_a: str = "stock_0",
    stock_b: str = "stock_73",
    short_window: int = 10,
    long_window: int = 50,
    entry_threshold: float = 2.0,
    exit_threshold: float = 1.0,
) -> PairStrategyResult:
    diagnostics = pair_diagnostics(returns, stock_a, stock_b)
    cumulative = cumulative_returns(returns)
    pair_spread = spread(cumulative, stock_a, stock_b)
    zscore = moving_average_zscore(pair_spread, short_window, long_window)
    signals = generate_rule_signals(zscore, entry_threshold, exit_threshold)
    spread_changes = pair_spread.diff()
    strategy_returns = signals.shift(1).fillna(0.0) * spread_changes
    strategy_returns.name = "strategy_return"

    return PairStrategyResult(
        stock_a=stock_a,
        stock_b=stock_b,
        spread=pair_spread,
        zscore=zscore,
        signals=signals,
        strategy_returns=strategy_returns,
        rolling_sharpe=rolling_sharpe(strategy_returns),
        diagnostics=diagnostics,
    )


def build_ml_pair_features(
    returns: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    cumulative = cumulative_returns(returns)
    pair_spread = spread(cumulative, stock_a, stock_b)
    short_ma = pair_spread.rolling(short_window, closed="left").mean()
    long_ma = pair_spread.rolling(long_window, closed="left").mean()
    ma_diff = short_ma - long_ma
    zscore = ma_diff / ma_diff.rolling(long_window).std()

    features = pd.DataFrame(
        {
            "zscore": zscore,
            "zscore_lag_1": zscore.shift(1),
            "short_ma": short_ma,
            "long_ma": long_ma,
            "spread_volatility": pair_spread.rolling(short_window, closed="left").std(),
            "spread_change_lag_1": pair_spread.diff().shift(1),
        }
    )
    return features.dropna()


def run_ml_pairs_strategy(
    returns: pd.DataFrame,
    stock_a: str = "stock_0",
    stock_b: str = "stock_73",
    short_window: int = 18,
    long_window: int = 81,
    entry_threshold: float = 2.14,
    exit_threshold: float = 0.94,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> PairStrategyResult:
    diagnostics = pair_diagnostics(returns, stock_a, stock_b)
    features = build_ml_pair_features(returns, stock_a, stock_b, short_window, long_window)
    labels = generate_rule_signals(features["zscore"], entry_threshold, exit_threshold).reindex(
        features.index
    )

    split = int(len(features) * train_fraction)
    x_train, x_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    train_score = float(accuracy_score(y_train, model.predict(x_train)))
    test_score = (
        float(accuracy_score(y_test, model.predict(x_test))) if len(x_test) else float("nan")
    )

    cumulative = cumulative_returns(returns)
    pair_spread = spread(cumulative, stock_a, stock_b)
    predictions = pd.Series(model.predict(features), index=features.index, name="signal")
    signals = predictions.reindex(pair_spread.index).ffill().fillna(0).astype(int)
    strategy_returns = signals.shift(1).fillna(0.0) * pair_spread.diff()
    strategy_returns.name = "strategy_return"

    return PairStrategyResult(
        stock_a=stock_a,
        stock_b=stock_b,
        spread=pair_spread,
        zscore=features["zscore"].reindex(pair_spread.index),
        signals=signals,
        strategy_returns=strategy_returns,
        rolling_sharpe=rolling_sharpe(strategy_returns),
        diagnostics=diagnostics,
        train_score=train_score,
        test_score=test_score,
    )
