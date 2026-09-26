import pandas as pd

from .blocking import block_pairs
from .normalize import normalize_columns


def generate_candidates(left: pd.DataFrame, right: pd.DataFrame, block_column: str) -> pd.DataFrame:
    """Normalize records and produce blocked candidate pairs."""
    left = normalize_columns(left, [block_column])
    right = normalize_columns(right, [block_column])
    return block_pairs(left, right, block_column)
