# Statistical Arbitrage Pairs Trading

This project refactors an exploratory take-home research notebook into a reproducible Python research repo. It studies two simple alpha ideas on anonymized equity data:

- a mean-reversion pairs strategy using cumulative-return spreads and moving-average z-scores
- an event-study strategy that estimates the average return path around binary events

The original assessment notebook is preserved in `notebooks/exploratory_research.ipynb`. The production-facing code lives in `src/stat_arb_pairs`.

## Repository Structure

```text
.
├── data/raw/                  # anonymized returns and event indicators
├── notebooks/                 # original exploratory notebook
├── src/stat_arb_pairs/        # reusable package code
├── tests/                     # focused unit tests
├── pyproject.toml             # package metadata and dependencies
└── README.md
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
stat-arb-backtest
```

List statistically screened candidate pairs:

```bash
stat-arb-backtest --list-pairs
```

Run a specific pairs backtest:

```bash
stat-arb-backtest \
  --stock-a stock_0 \
  --stock-b stock_73 \
  --short-window 18 \
  --long-window 81 \
  --entry-threshold 2.14
```

## Methodology

The pairs strategy:

1. Converts daily stock returns to cumulative returns.
2. Defines a spread between two stocks.
3. Computes short- and long-window moving averages of the spread.
4. Converts the moving-average difference into a rolling z-score.
5. Enters long-spread positions when the z-score is below the negative entry threshold and short-spread positions when it is above the positive entry threshold.
6. Exits when the absolute z-score falls below the exit threshold.

The event strategy:

1. Finds complete windows around each event date for a selected stock.
2. Estimates the average event-centered return path.
3. Trades in the direction of the average post-event drift for a fixed holding period.

## Current Limitations

This is a research project, not a production trading system. Important next steps before treating results as economically meaningful:

- add transaction costs, borrow constraints, and slippage
- use strict train/test splits for parameter selection
- avoid optimizing thresholds on the same sample used for reporting
- add richer risk reporting and exposure controls
- expand event-model validation beyond average path direction

## Data

The included data uses anonymized stock identifiers (`stock_0`, `stock_1`, etc.). No ticker mapping is assumed by the code.

