"""Reusable research components for the statistical arbitrage project."""

from stat_arb_pairs.data import load_events, load_returns
from stat_arb_pairs.events import EventStudyStrategy
from stat_arb_pairs.metrics import annualized_sharpe, cumulative_returns, max_drawdown
from stat_arb_pairs.pairs import PairCandidate, PairsTradingStrategy, find_candidate_pairs

__all__ = [
    "EventStudyStrategy",
    "PairCandidate",
    "PairsTradingStrategy",
    "annualized_sharpe",
    "cumulative_returns",
    "find_candidate_pairs",
    "load_events",
    "load_returns",
    "max_drawdown",
]

