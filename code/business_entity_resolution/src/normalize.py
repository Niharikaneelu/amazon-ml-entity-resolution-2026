import re
import unicodedata

import pandas as pd


_BUSINESS_SUFFIXES = {
    "ag",
    "bv",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
    "pty",
    "sarl",
}

_ADDRESS_ABBREVIATIONS = {
    "avenue": "ave",
    "boulevard": "blvd",
    "circle": "cir",
    "court": "ct",
    "drive": "dr",
    "highway": "hwy",
    "lane": "ln",
    "parkway": "pkwy",
    "place": "pl",
    "road": "rd",
    "square": "sq",
    "street": "st",
    "terrace": "ter",
    "trail": "trl",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "room": "rm",
    "suite": "ste",
    "north": "n",
    "northeast": "ne",
    "northwest": "nw",
    "south": "s",
    "southeast": "se",
    "southwest": "sw",
    "east": "e",
    "west": "w",
}


def _clean_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).split()


def normalize_business_name(value: object) -> str:
    """Normalize a business name without changing the original value."""
    tokens = _clean_tokens(value)
    while tokens and tokens[-1] in _BUSINESS_SUFFIXES:
        tokens.pop()
    if tokens and tokens[-1] == "and":
        tokens.pop()
    return " ".join(tokens)


def normalize_business_address(value: object) -> str:
    """Normalize an address and standardize common street and unit terms."""
    tokens = _clean_tokens(value)
    return " ".join(_ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens)


def normalize_business_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the required normalized business columns added."""
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = frame.copy()
    result["business_name_norm"] = result["business_name"].map(normalize_business_name)
    result["business_address_norm"] = result["business_address"].map(normalize_business_address)
    return result


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
