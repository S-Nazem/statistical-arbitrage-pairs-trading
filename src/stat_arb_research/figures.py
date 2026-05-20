from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stat_arb_research.data import load_events, load_stock_returns
from stat_arb_research.events import fit_event_study, run_event_strategy
from stat_arb_research.metrics import cumulative_returns
from stat_arb_research.pairs import (
    find_correlated_pairs,
    moving_average_zscore,
    pair_diagnostics,
    run_ml_pairs_strategy,
    run_rule_based_pairs_strategy,
    spread,
)

FIGURE_DIR = Path("reports") / "figures"
ACCENT = "#1f77b4"
BUY = "#218c74"
SELL = "#c44536"
MUTED = "#6c757d"


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_signal_overlay(ax: plt.Axes, series: pd.Series, signals: pd.Series) -> None:
    aligned = signals.reindex(series.index).fillna(0)
    ax.plot(series.index, series, color="#222222", linewidth=1.2, label=series.name or "Spread")
    buys = series[aligned == 1]
    sells = series[aligned == -1]
    ax.scatter(buys.index, buys, color=BUY, marker="^", s=18, label="Long spread", zorder=3)
    ax.scatter(sells.index, sells, color=SELL, marker="v", s=18, label="Short spread", zorder=3)


def create_all_figures(output_dir: str | Path = FIGURE_DIR) -> dict[str, float]:
    _setup_style()
    output_dir = Path(output_dir)
    returns = load_stock_returns()
    events = load_events()
    cumulative = cumulative_returns(returns)
    candidates = find_correlated_pairs(returns, base_stock="stock_0", top_n=4)
    candidate_stocks = candidates["stock_b"].tolist()

    metrics: dict[str, float] = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(returns.mean(), bins=30, color=ACCENT, alpha=0.82, edgecolor="white")
    axes[0].axvline(returns.mean().mean(), color="#222222", linestyle="--", linewidth=1)
    axes[0].set_title("Cross-Sectional Mean Daily Returns")
    axes[0].set_xlabel("Mean daily return")
    axes[0].set_ylabel("Number of stocks")
    axes[1].hist(returns.std(), bins=30, color="#d97706", alpha=0.82, edgecolor="white")
    axes[1].axvline(returns.std().mean(), color="#222222", linestyle="--", linewidth=1)
    axes[1].set_title("Cross-Sectional Daily Volatility")
    axes[1].set_xlabel("Daily volatility")
    _save(fig, output_dir / "01_return_distribution_summary.png")

    top_10 = cumulative.max().sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative[top_10].plot(ax=ax, linewidth=1.1, alpha=0.85)
    ax.set_title("Top 10 Stocks by Peak Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(ncol=2, frameon=False)
    _save(fig, output_dir / "02_top_10_cumulative_returns.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, stock_b in zip(axes.ravel(), candidate_stocks):
        ax.plot(cumulative.index, cumulative["stock_0"], label="stock_0", color="#222222")
        ax.plot(cumulative.index, cumulative[stock_b], label=stock_b, color=ACCENT)
        diag = pair_diagnostics(returns, "stock_0", stock_b)
        ax.set_title(f"stock_0 vs {stock_b} | corr={diag.correlation:.3f}")
        ax.set_ylabel("Cumulative return")
        ax.legend(frameon=False)
    _save(fig, output_dir / "03_candidate_pair_cumulative_returns.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, stock_b in zip(axes.ravel(), candidate_stocks):
        pair_spread = spread(cumulative, "stock_0", stock_b)
        ax.plot(pair_spread.index, pair_spread, color=ACCENT, linewidth=1)
        ax.axhline(pair_spread.mean(), color=SELL, linestyle="--", linewidth=1, label="Mean spread")
        ax.set_title(f"Spread: stock_0 - {stock_b}")
        ax.set_ylabel("Spread")
        ax.legend(frameon=False)
    _save(fig, output_dir / "04_candidate_pair_spreads.png")

    rule = run_rule_based_pairs_strategy(returns, stock_a="stock_0", stock_b="stock_73")
    metrics["rule_pairs_sharpe"] = rule.sharpe
    metrics["pair_correlation"] = rule.diagnostics.correlation
    metrics["pair_spread_adf_pvalue"] = rule.diagnostics.spread_adf_pvalue

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_signal_overlay(ax, rule.spread.rename("Spread"), rule.signals)
    ax.set_title(f"Rule-Based Pairs Signals | Sharpe={rule.sharpe:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative-return spread")
    ax.legend(frameon=False, ncol=3)
    _save(fig, output_dir / "05_rule_based_pairs_signals.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rule.rolling_sharpe.index, rule.rolling_sharpe, color=ACCENT)
    ax.axhline(1, color=BUY, linestyle="--", linewidth=1)
    ax.axhline(-1, color=SELL, linestyle="--", linewidth=1)
    ax.set_title("Rule-Based Pairs Strategy: 252-Day Rolling Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    _save(fig, output_dir / "06_rule_based_pairs_rolling_sharpe.png")

    ml = run_ml_pairs_strategy(returns, stock_a="stock_0", stock_b="stock_73")
    metrics["ml_pairs_sharpe"] = ml.sharpe
    metrics["ml_pairs_train_score"] = ml.train_score or float("nan")
    metrics["ml_pairs_test_score"] = ml.test_score or float("nan")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_signal_overlay(ax, ml.spread.rename("Spread"), ml.signals)
    split = int(len(ml.signals.dropna()) * 0.8)
    split_date = ml.signals.index[split]
    ax.axvline(split_date, color=MUTED, linestyle="--", linewidth=1, label="Train/test split")
    ax.set_title(
        "Random-Forest Pairs Signals | "
        f"Sharpe={ml.sharpe:.2f}, test accuracy={ml.test_score:.2f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative-return spread")
    ax.legend(frameon=False, ncol=4)
    _save(fig, output_dir / "07_ml_pairs_signals.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ml.rolling_sharpe.index, ml.rolling_sharpe, color=ACCENT)
    ax.axhline(1, color=BUY, linestyle="--", linewidth=1)
    ax.axhline(-1, color=SELL, linestyle="--", linewidth=1)
    ax.set_title("Random-Forest Pairs Strategy: 252-Day Rolling Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    _save(fig, output_dir / "08_ml_pairs_rolling_sharpe.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pair_spread = spread(cumulative, "stock_0", "stock_73")
    zscore = moving_average_zscore(pair_spread, 18, 81)
    ax.plot(zscore.index, zscore, color=ACCENT, linewidth=1)
    ax.axhline(2.14, color=SELL, linestyle="--", linewidth=1, label="Entry threshold")
    ax.axhline(-2.14, color=BUY, linestyle="--", linewidth=1)
    ax.axhspan(-0.94, 0.94, color="#adb5bd", alpha=0.25, label="Exit zone")
    ax.set_title("Pairs Signal Feature: Moving-Average Spread Z-Score")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-score")
    ax.legend(frameon=False)
    _save(fig, output_dir / "09_pairs_zscore_thresholds.png")

    study = fit_event_study(returns, events, stock="stock_1", window=5)
    metrics["event_study_rmse"] = study.rmse
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for _, row in study.event_windows.iterrows():
        ax.plot(study.mean_path.index, row, color="#8d99ae", alpha=0.22, linewidth=0.8)
    ax.plot(
        study.mean_path.index,
        study.mean_path,
        color="#111111",
        linewidth=2.2,
        label="Mean path",
    )
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title("Event Study: Normalized Return Paths Around Events")
    ax.set_xlabel("Business days relative to event")
    ax.set_ylabel("Cumulative return relative to event day")
    ax.legend(frameon=False)
    _save(fig, output_dir / "10_event_study_return_paths.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(study.mean_path.index, study.mean_path, color="#111111", linewidth=2, label="Mean path")
    ax.plot(
        study.fitted_path.index,
        study.fitted_path,
        color=SELL,
        linestyle="--",
        label="Quadratic fit",
    )
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title(f"Event Study Quadratic Fit | RMSE={study.rmse:.4f}")
    ax.set_xlabel("Business days relative to event")
    ax.set_ylabel("Normalized cumulative return")
    ax.legend(frameon=False)
    _save(fig, output_dir / "11_event_study_polynomial_fit.png")

    event_strategy = run_event_strategy(returns, events, stock="stock_1", window=5)
    metrics["event_strategy_sharpe"] = float(event_strategy["sharpe"])
    cumulative_stock = cumulative["stock_1"]
    signals = event_strategy["signals"]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_signal_overlay(ax, cumulative_stock.rename("stock_1 cumulative return"), signals)
    ax.set_title(f"Rule-Based Event Signals | Sharpe={event_strategy['sharpe']:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(frameon=False, ncol=3)
    _save(fig, output_dir / "12_event_strategy_signals.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    rolling = event_strategy["rolling_sharpe"]
    ax.plot(rolling.index, rolling, color=ACCENT)
    ax.axhline(1, color=BUY, linestyle="--", linewidth=1)
    ax.axhline(-1, color=SELL, linestyle="--", linewidth=1)
    ax.set_title("Rule-Based Event Strategy: 252-Day Rolling Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    _save(fig, output_dir / "13_event_strategy_rolling_sharpe.png")

    return metrics
