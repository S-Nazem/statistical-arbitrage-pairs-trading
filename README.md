# Statistical Arbitrage Research

This repository turns an exploratory quant research notebook into a small, reproducible Python project. The work studies two simple alpha ideas on anonymized equity data:

- mean-reversion pairs trading using cumulative-return spreads
- event-study trading around binary stock-level events

The pairs strategy is the main research thread. The event strategy is included as an exploratory extension and is reported more cautiously because the backtest is weaker.

## Project Shape

```text
.
├── data/raw/                         # anonymized returns and event inputs
├── notebooks/exodus_research_test.ipynb
├── reports/figures/                  # generated research figures
├── src/stat_arb_research/            # reusable research code
├── tests/                            # focused unit tests
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pytest
stat-arb-research --figures
```

The figure command regenerates the images in `reports/figures` directly from the raw CSV files.

## Data

The assessment data is anonymized:

- `stock_returns.csv`: daily returns for `stock_0` through `stock_99`
- `events.csv`: binary event flags for the same stocks and dates

No ticker mapping or external market data is assumed.

## Methodology

### Pairs Trading

The pairs workflow follows the notebook's core idea:

1. Convert daily returns to cumulative returns as a proxy for normalized price paths.
2. Search for highly correlated stock pairs.
3. Check whether the pair spread is stationary using ADF and cointegration tests.
4. Build the spread between the cumulative-return paths.
5. Compare short- and long-window moving averages of the spread.
6. Generate positions when the moving-average spread z-score crosses entry and exit thresholds.
7. Report strategy returns and 252-day rolling Sharpe.

The random-forest version uses lagged spread features and chronological train/test splits. Signals are shifted by one period before returns are calculated to avoid the most direct form of same-period lookahead.

### Event Study

The event workflow:

1. Extract windows around each event date for a selected stock.
2. Normalize each path to zero on the event date.
3. Estimate the average event response.
4. Fit a quadratic model to the average path.
5. Use the fitted turning point to generate simple rule-based event signals.

This section is useful as a research extension, but it is less convincing than the pairs strategy in its current form.

## Headline Findings

The notebook found a strong candidate pair between `stock_0` and `stock_73`, and the refactored code keeps that pair as the default demonstration case.

Current regenerated metrics:

- pair cumulative-return correlation: `0.9747`
- pair spread ADF p-value: `0.0006`
- rule-based pairs Sharpe: `0.56`
- random-forest pairs Sharpe: `0.21`
- random-forest pairs test accuracy: `0.66`
- event-study quadratic fit RMSE: `0.0060`
- rule-based event strategy Sharpe: `-0.11`

The ML pairs model should be read as a prototype, not a production alpha model. It learns from rule-generated labels and still needs stricter validation before the result could be considered economically meaningful.

## Limitations

This is a research backtest, not a deployable trading system. Important missing pieces include:

- transaction costs, borrow costs, slippage, and liquidity constraints
- walk-forward parameter selection
- stronger out-of-sample validation
- market and sector exposure controls
- benchmark-relative risk reporting
- robustness checks across many pairs and event types

## Original Notebook

The original exploratory assessment notebook is preserved at `notebooks/exodus_research_test.ipynb`. The package code is intentionally cleaner and more conservative than the raw notebook so the repo can serve as a portfolio-ready research artifact.
