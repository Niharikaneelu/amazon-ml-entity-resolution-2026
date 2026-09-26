import re
import unicodedata

import pandas as pd


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[f"{column}__norm"] = result[column].map(normalize_text)
    return result
