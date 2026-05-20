"""Research utilities for the statistical arbitrage portfolio project."""

from stat_arb_research.data import load_events, load_stock_returns
from stat_arb_research.events import EventStudyResult, fit_event_study, run_event_strategy
from stat_arb_research.metrics import annualized_sharpe, cumulative_returns, rolling_sharpe
from stat_arb_research.pairs import (
    PairDiagnostics,
    PairStrategyResult,
    find_correlated_pairs,
    run_ml_pairs_strategy,
    run_rule_based_pairs_strategy,
)

__all__ = [
    "EventStudyResult",
    "PairDiagnostics",
    "PairStrategyResult",
    "annualized_sharpe",
    "cumulative_returns",
    "find_correlated_pairs",
    "fit_event_study",
    "load_events",
    "load_stock_returns",
    "rolling_sharpe",
    "run_event_strategy",
    "run_ml_pairs_strategy",
    "run_rule_based_pairs_strategy",
]
