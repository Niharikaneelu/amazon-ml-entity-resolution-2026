import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

REQUIRED_FEATURES = [
    "name_exact",
    "name_similarity",
    "name_token_similarity",
    "address_exact",
    "address_similarity",
    "address_token_similarity",
    "country_match",
    "house_number_match",
    "name_missing",
    "address_missing",
]

EXTENDED_FEATURES = [
    "name_jaro_winkler",
    "name_token_jaccard",
    "is_source2",
    "is_source3",
]

FEATURE_COLUMNS = REQUIRED_FEATURES + EXTENDED_FEATURES


def _clean_string(value: object) -> str:
    """Normalize string to ascii lowercase with normalized whitespace."""
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", norm).strip()


def _string_similarity(s1: str, s2: str) -> float:
    """Compute string similarity ratio between 0.0 and 1.0."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.ratio(s1, s2) / 100.0)
    return float(SequenceMatcher(None, s1, s2).ratio())


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler string similarity between 0.0 and 1.0."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(JaroWinkler.similarity(s1, s2))
    return float(SequenceMatcher(None, s1, s2).ratio())


def _token_similarity(s1: str, s2: str) -> float:
    """Compute token sort/set similarity between 0.0 and 1.0."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_sort_ratio(s1, s2) / 100.0)

    # Fallback to Jaccard similarity of alphanumeric tokens
    tokens1 = set(re.findall(r"\b[a-z0-9]+\b", s1))
    tokens2 = set(re.findall(r"\b[a-z0-9]+\b", s2))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection / union) if union > 0 else 0.0


def _token_jaccard(s1: str, s2: str) -> float:
    """Compute Jaccard token similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    tok1 = set(re.findall(r"\b[a-z0-9]+\b", s1))
    tok2 = set(re.findall(r"\b[a-z0-9]+\b", s2))
    if not tok1 or not tok2:
        return 0.0
    return float(len(tok1 & tok2) / len(tok1 | tok2))


def _extract_numbers(text: str) -> set[str]:
    """Extract sequences of digits representing house numbers or street numbers."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def _resolve_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def merge_entity_attributes(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge entity attributes (business_name, business_address, country) onto pairs.

    Args:
        pairs: DataFrame containing source1_entity_id and candidate_entity_id.
        s1_df: DataFrame containing source 1 entities.
        cand_df: DataFrame or mapping/list of candidate entities (source2, source3).

    Returns:
        DataFrame with pair IDs and paired attribute columns with _s1 and _cand suffixes.
    """
    result = pairs.copy()

    s1_cols = [c for c in ["business_name", "business_address", "country"] if c in s1_df.columns]
    s1_subset = s1_df[["entity_id"] + s1_cols].drop_duplicates(subset=["entity_id"])
    s1_rename = {"entity_id": "source1_entity_id"}
    for col in s1_cols:
        s1_rename[col] = f"{col}_s1"
    s1_subset = s1_subset.rename(columns=s1_rename)

    if isinstance(cand_df, dict):
        cand_combined = pd.concat(list(cand_df.values()), ignore_index=True)
    elif isinstance(cand_df, list):
        cand_combined = pd.concat(cand_df, ignore_index=True)
    else:
        cand_combined = cand_df

    cand_cols = [c for c in ["business_name", "business_address", "country"] if c in cand_combined.columns]
    cand_subset = cand_combined[["entity_id"] + cand_cols].drop_duplicates(subset=["entity_id"])
    cand_rename = {"entity_id": "candidate_entity_id"}
    for col in cand_cols:
        cand_rename[col] = f"{col}_cand"
    cand_subset = cand_subset.rename(columns=cand_rename)

    result = result.merge(s1_subset, on="source1_entity_id", how="left")
    result = result.merge(cand_subset, on="candidate_entity_id", how="left")
    return result


def build_matching_features(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame | None = None,
    cand_df: pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build entity resolution matching features for candidate pairs.

    Features generated:
        - name_exact: 1.0 if normalized business names match identically, else 0.0
        - name_similarity: String similarity ratio between business names (0.0 to 1.0)
        - name_token_similarity: Token-based similarity between business names (0.0 to 1.0)
        - address_exact: 1.0 if normalized addresses match identically, else 0.0
        - address_similarity: String similarity ratio between addresses (0.0 to 1.0)
        - address_token_similarity: Token-based similarity between addresses (0.0 to 1.0)
        - country_match: 1.0 if countries match, else 0.0
        - house_number_match: 1.0 if address numbers match, else 0.0
        - name_missing: 1.0 if either name is missing/empty, else 0.0
        - address_missing: 1.0 if either address is missing/empty, else 0.0
        - name_jaro_winkler: Jaro-Winkler similarity on business names
        - name_token_jaccard: Jaccard word token overlap ratio
        - is_source2: 1.0 if candidate is from Source 2, else 0.0
        - is_source3: 1.0 if candidate is from Source 3, else 0.0

    Args:
        pairs: DataFrame containing pair identifiers or merged attributes.
        s1_df: Optional Source 1 DataFrame to merge attributes if not present.
        cand_df: Optional candidate DataFrame(s) to merge attributes if not present.

    Returns:
        DataFrame with feature columns indexed matching `pairs`.
    """
    df = pairs
    if s1_df is not None and cand_df is not None:
        if "source1_entity_id" in pairs.columns and "candidate_entity_id" in pairs.columns:
            df = merge_entity_attributes(pairs, s1_df, cand_df)

    col_name_s1 = _resolve_column(df, [
        "business_name_s1", "business_name_left", "source1_business_name",
        "name_left", "name_s1"
    ])
    col_name_cand = _resolve_column(df, [
        "business_name_cand", "business_name_right", "candidate_business_name",
        "name_right", "name_cand"
    ])

    col_addr_s1 = _resolve_column(df, [
        "business_address_s1", "business_address_left", "source1_business_address",
        "address_left", "address_s1"
    ])
    col_addr_cand = _resolve_column(df, [
        "business_address_cand", "business_address_right", "candidate_business_address",
        "address_right", "address_cand"
    ])

    col_country_s1 = _resolve_column(df, [
        "country_s1", "country_left", "source1_country", "country"
    ])
    col_country_cand = _resolve_column(df, [
        "country_cand", "country_right", "candidate_country"
    ])

    n_rows = len(df)
    names_1 = [_clean_string(v) for v in df[col_name_s1]] if col_name_s1 else [""] * n_rows
    names_2 = [_clean_string(v) for v in df[col_name_cand]] if col_name_cand else [""] * n_rows

    addrs_1 = [_clean_string(v) for v in df[col_addr_s1]] if col_addr_s1 else [""] * n_rows
    addrs_2 = [_clean_string(v) for v in df[col_addr_cand]] if col_addr_cand else [""] * n_rows

    cntrys_1 = [_clean_string(v) for v in df[col_country_s1]] if col_country_s1 else [""] * n_rows
    cntrys_2 = [_clean_string(v) for v in df[col_country_cand]] if col_country_cand else [""] * n_rows

    name_exact = []
    name_sim = []
    name_tok_sim = []
    name_missing = []
    name_jw = []
    name_jaccard = []

    for n1, n2 in zip(names_1, names_2):
        is_missing = float(not n1 or not n2)
        name_missing.append(is_missing)
        if not n1 or not n2:
            name_exact.append(0.0)
            name_sim.append(0.0)
            name_tok_sim.append(0.0)
            name_jw.append(0.0)
            name_jaccard.append(0.0)
        else:
            name_exact.append(float(n1 == n2))
            name_sim.append(_string_similarity(n1, n2))
            name_tok_sim.append(_token_similarity(n1, n2))
            name_jw.append(_jaro_winkler_similarity(n1, n2))
            name_jaccard.append(_token_jaccard(n1, n2))

    addr_exact = []
    addr_sim = []
    addr_tok_sim = []
    addr_missing = []
    house_num_match = []

    for a1, a2 in zip(addrs_1, addrs_2):
        is_missing = float(not a1 or not a2)
        addr_missing.append(is_missing)
        if not a1 or not a2:
            addr_exact.append(0.0)
            addr_sim.append(0.0)
            addr_tok_sim.append(0.0)
            house_num_match.append(0.0)
        else:
            addr_exact.append(float(a1 == a2))
            addr_sim.append(_string_similarity(a1, a2))
            addr_tok_sim.append(_token_similarity(a1, a2))
            nums1 = _extract_numbers(a1)
            nums2 = _extract_numbers(a2)
            house_num_match.append(float(bool(nums1 and nums2 and (nums1 & nums2))))

    country_match = []
    for c1, c2 in zip(cntrys_1, cntrys_2):
        if not c1 or not c2:
            country_match.append(0.0)
        else:
            country_match.append(float(c1 == c2))

    # Candidate source features
    is_source2 = []
    is_source3 = []
    if "candidate_source" in df.columns:
        for s in df["candidate_source"]:
            s_str = str(s).lower()
            is_source2.append(float("source2" in s_str or "s2" in s_str))
            is_source3.append(float("source3" in s_str or "s3" in s_str))
    elif "candidate_entity_id" in df.columns:
        for cid in df["candidate_entity_id"]:
            cid_str = str(cid).upper()
            is_source2.append(float(cid_str.startswith("S2")))
            is_source3.append(float(cid_str.startswith("S3")))
    else:
        is_source2 = [0.0] * n_rows
        is_source3 = [0.0] * n_rows

    features = pd.DataFrame(
        {
            "name_exact": name_exact,
            "name_similarity": name_sim,
            "name_token_similarity": name_tok_sim,
            "address_exact": addr_exact,
            "address_similarity": addr_sim,
            "address_token_similarity": addr_tok_sim,
            "country_match": country_match,
            "house_number_match": house_num_match,
            "name_missing": name_missing,
            "address_missing": addr_missing,
            "name_jaro_winkler": name_jw,
            "name_token_jaccard": name_jaccard,
            "is_source2": is_source2,
            "is_source3": is_source3,
        },
        index=pairs.index,
    )
    return features


def build_features(
    pairs: pd.DataFrame,
    columns: list[str] | None = None,
    s1_df: pd.DataFrame | None = None,
    cand_df: pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build feature set for candidate pairs.

    If candidate pair schema or attribute columns are present, builds the standard
    matching feature suite (name_exact, name_similarity, address_similarity, etc.).
    Maintains backward compatibility with field-level similarity lists.
    """
    col_name_s1 = _resolve_column(pairs, [
        "business_name_s1", "business_name_left", "source1_business_name",
        "name_left", "name_s1"
    ])
    if col_name_s1 is not None or (s1_df is not None and cand_df is not None):
        return build_matching_features(pairs, s1_df=s1_df, cand_df=cand_df)

    if columns is not None:
        features = {}
        for column in columns:
            left = f"{column}_left"
            right = f"{column}_right"
            if left not in pairs or right not in pairs:
                continue
            features[f"{column}_similarity"] = [
                _string_similarity(_clean_string(a), _clean_string(b))
                for a, b in zip(pairs[left], pairs[right])
            ]
        result = pd.DataFrame(features, index=pairs.index)
        if result.empty:
            raise ValueError("No matching field columns were found in candidate pairs")
        result["mean_similarity"] = result.mean(axis=1)
        return result

    # Default fallback: attempt matching features
    return build_matching_features(pairs, s1_df=s1_df, cand_df=cand_df)
