from __future__ import annotations

import argparse

from stat_arb_pairs.data import load_events, load_returns
from stat_arb_pairs.events import EventStudyStrategy
from stat_arb_pairs.pairs import PairsTradingStrategy, find_candidate_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run statistical arbitrage backtests.")
    parser.add_argument("--returns", default="data/raw/stock_returns.csv")
    parser.add_argument("--events", default="data/raw/events.csv")
    parser.add_argument("--stock-a", default="stock_0")
    parser.add_argument("--stock-b", default="stock_73")
    parser.add_argument("--event-stock", default="stock_1")
    parser.add_argument("--short-window", type=int, default=18)
    parser.add_argument("--long-window", type=int, default=81)
    parser.add_argument("--entry-threshold", type=float, default=2.14)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--list-pairs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    returns = load_returns(args.returns)
    events = load_events(args.events)

    if args.list_pairs:
        for pair in find_candidate_pairs(returns):
            print(
                f"{pair.stock_a}/{pair.stock_b}: "
                f"corr={pair.correlation:.3f}, coint_p={pair.cointegration_pvalue:.4f}"
            )
        return

    pairs_result = PairsTradingStrategy(
        args.stock_a,
        args.stock_b,
        short_window=args.short_window,
        long_window=args.long_window,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
    ).backtest(returns)

    event_result = EventStudyStrategy(args.event_stock).backtest(returns, events)

    print("Pairs strategy")
    print(f"  pair: {args.stock_a}/{args.stock_b}")
    print(f"  sharpe: {pairs_result.sharpe:.3f}")
    print(f"  max_drawdown: {pairs_result.max_drawdown:.3%}")
    print()
    print("Event strategy")
    print(f"  stock: {args.event_stock}")
    print(f"  sharpe: {event_result.sharpe:.3f}")
    print(f"  max_drawdown: {event_result.max_drawdown:.3%}")


if __name__ == "__main__":
    main()

