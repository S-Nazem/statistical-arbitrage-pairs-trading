from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETURNS_PATH = PROJECT_ROOT / "data" / "raw" / "stock_returns.csv"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "events.csv"
PathLike = Union[str, Path]


def _load_time_indexed_csv(path: PathLike) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame.columns = frame.columns.astype(str)
    return frame


def load_returns(path: PathLike = DEFAULT_RETURNS_PATH) -> pd.DataFrame:
    """Load daily stock returns with a DatetimeIndex."""
    returns = _load_time_indexed_csv(path)
    if returns.isna().any().any():
        raise ValueError("Returns data contains missing values.")
    return returns


def load_events(path: PathLike = DEFAULT_EVENTS_PATH) -> pd.DataFrame:
    """Load binary event indicators with a DatetimeIndex."""
    events = _load_time_indexed_csv(path)
    invalid = ~events.isin([0, 1])
    if invalid.any().any():
        raise ValueError("Events data must contain only 0/1 indicators.")
    return events.astype(int)
