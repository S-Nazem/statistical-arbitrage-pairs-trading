from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame.sort_index()


def load_stock_returns(path: str | Path = RAW_DATA_DIR / "stock_returns.csv") -> pd.DataFrame:
    """Load the anonymized daily stock-return matrix."""

    return _load_csv(path).astype(float)


def load_events(path: str | Path = RAW_DATA_DIR / "events.csv") -> pd.DataFrame:
    """Load the anonymized binary event matrix."""

    return _load_csv(path).astype(int)
