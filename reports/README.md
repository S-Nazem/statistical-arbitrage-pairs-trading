# Results Worklist

The current project has reusable strategy code and recovered notebook figures. To make the results presentation repo-ready, the next work should be:

1. Re-run all reported backtests from package code rather than notebook state.
2. Add a single `results.csv` with strategy name, parameters, Sharpe, max drawdown, annualized return, volatility, hit rate, and turnover.
3. Generate clean, consistently styled figures from scripts instead of embedded notebook outputs.
4. Split results into train/test periods so optimized parameters are not evaluated on the same sample used to choose them.
5. Add transaction costs and slippage assumptions.
6. Add benchmark comparisons, such as equal-weight long-only returns and random-event baselines.
7. Write a short `reports/results.md` explaining methodology, assumptions, main findings, and limitations.

