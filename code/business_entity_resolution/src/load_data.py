from __future__ import annotations
from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV and fail with a useful message when it is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
