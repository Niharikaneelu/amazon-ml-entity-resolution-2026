import pandas as pd


def block_pairs(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """Create candidate pairs sharing a normalized blocking key."""
    key = f"{key}__norm"
    if key not in left or key not in right:
        raise ValueError(f"Blocking column must exist on both frames: {key}")
    candidates = left.merge(right, on=key, suffixes=("_left", "_right"))
    return candidates.drop_duplicates()
