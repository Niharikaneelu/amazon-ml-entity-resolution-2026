from pathlib import Path
import pandas as pd


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV/TSV table and fail with a useful message when absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    sep = "\t" if path.suffix.lower() == ".tsv" or "tsv" in path.name.lower() else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8", nrows=nrows)


def read_tsv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a tab-separated TSV file explicitly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep="\t", encoding="utf-8", nrows=nrows)



def validate_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

