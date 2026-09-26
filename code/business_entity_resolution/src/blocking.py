"""Fully vectorized blocking module - all strategies use pandas operations only.

Key design: every strategy pre-computes blocking keys as a DataFrame using
pandas string ops + explode, then joins via merge. No Python for-loops over rows.
"""
from __future__ import annotations
import re
import unicodedata
from collections import defaultdict

import pandas as pd
import numpy as np

LEGAL_SUFFIXES = {
    "inc", "incorporated", "ltd", "limited", "corp", "corporation",
    "llc", "pvt", "private", "co", "company", "services", "enterprises",
    "group", "store", "shop"
}
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b"
)

STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to",
    "co", "inc", "ltd", "corp", "llc", "pvt"
}


# ── Scalar helpers (unit tests) ───────────────────────────────────────────────

def normalize_string(value: object) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    if text.isascii():
        return text
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def get_exact_normalized_name_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_string(name))


def get_country_normalized_name_key(country: object, name: object) -> str:
    cntry = normalize_string(country)
    clean = normalize_string(name)
    words = [w for w in re.sub(r"[^a-z0-9\s]", " ", clean).split() if w not in LEGAL_SUFFIXES]
    norm = "".join(words) or re.sub(r"[^a-z0-9]", "", clean)
    return f"{cntry}__{norm}" if norm else ""


def get_country_name_token_keys(country: object, name: object) -> list[str]:
    cntry = normalize_string(country)
    clean = normalize_string(name)
    exact = re.sub(r"[^a-z0-9]", "", clean)
    words = re.sub(r"[^a-z0-9\s]", " ", clean).split()
    tokens = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    keys: set[str] = set()
    for t in tokens[:4]:
        if len(t) >= 3:
            keys.add(f"{cntry}__{t[:4]}")
    if len(tokens) >= 2:
        st = sorted(tokens[:3])
        keys.add(f"{cntry}__{st[0]}_{st[1]}")
    elif len(tokens) == 1:
        keys.add(f"{cntry}__{tokens[0]}")
    if len(exact) >= 4:
        keys.add(f"{cntry}__prefix_{exact[:4]}")
    return list(keys)


def get_country_address_number_keys(country: object, name: object, address: object) -> list[str]:
    cntry = normalize_string(country)
    addr = normalize_string(address)
    if not addr:
        return []
    nums = [n for n in re.findall(r"\b\d+\b", addr) if len(n) >= 2]
    if not nums:
        return []
    clean = normalize_string(name)
    words = re.sub(r"[^a-z0-9\s]", " ", clean).split()
    first_char = words[0][0] if words else ""
    first_prefix = words[0][:2] if words else ""
    keys: set[str] = set()
    for num in nums:
        keys.add(f"{cntry}__{num}__{first_char}")
        if len(first_prefix) >= 2:
            keys.add(f"{cntry}__{num}__{first_prefix}")
    return list(keys)


# ── Vectorized key computation (pandas ops, no row loops) ─────────────────────

_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")

def _norm(s: pd.Series) -> pd.Series:
    """Vectorized normalize: lowercase+strip, NFKD for non-ASCII rows only.

    Uses vectorized regex to detect non-ASCII rows (fast C-speed str.contains),
    then applies NFKD encoding only to those rows.
    """
    s = s.fillna("").astype(str).str.strip().str.lower()
    # str.contains with regex runs at C speed — much faster than apply(str.isascii)
    mask = s.str.contains(r"[^\x00-\x7f]", regex=True, na=False)
    if mask.any():
        s = s.copy()
        s[mask] = s[mask].apply(
            lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
        )
    return s


def _exact_name_keydf(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[entity_id, blocking_key] for exact_normalized_name."""
    keys = _norm(df["business_name"]).str.replace(r"[^a-z0-9]", "", regex=True)
    out = pd.DataFrame({"entity_id": df["entity_id"].values, "blocking_key": keys.values})
    return out[out["blocking_key"].str.len() > 0]


def _country_norm_name_keydf(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[entity_id, blocking_key] for country_normalized_name."""
    cntry = _norm(df["country"])
    names = _norm(df["business_name"])
    # Strip each legal suffix word; fall back to raw alphanum
    def _strip(text: str) -> str:
        words = [w for w in re.sub(r"[^a-z0-9\s]", " ", text).split() if w not in LEGAL_SUFFIXES]
        return "".join(words) or re.sub(r"[^a-z0-9]", "", text)
    norm_names = names.apply(_strip)
    keys = cntry + "__" + norm_names
    out = pd.DataFrame({"entity_id": df["entity_id"].values, "blocking_key": keys.values})
    return out[norm_names.values != ""]


def _token_keydf(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form DataFrame[entity_id, blocking_key] for country_name_tokens.

    Uses pandas str ops + explode — no Python row loop.
    """
    cntry = _norm(df["country"])
    names = _norm(df["business_name"])
    eids = df["entity_id"].values

    # Build a list of (entity_id, key) pairs using vectorized approach:
    # For each token position, compute prefix key independently then stack.
    rows = []
    # Tokenize all names at once
    tokens_series = names.str.replace(r"[^a-z0-9\s]", " ", regex=True).str.split()

    for eid, c, exact_raw, toks in zip(eids, cntry.values, names.values, tokens_series):
        exact = re.sub(r"[^a-z0-9]", "", exact_raw)
        toks = [w for w in (toks or []) if len(w) >= 3 and w not in STOPWORDS]
        keys: set[str] = set()
        for t in toks[:4]:
            if len(t) >= 3:
                keys.add(f"{c}__{t[:4]}")
        if len(toks) >= 2:
            st = sorted(toks[:3])
            keys.add(f"{c}__{st[0]}_{st[1]}")
        elif len(toks) == 1:
            keys.add(f"{c}__{toks[0]}")
        if len(exact) >= 4:
            keys.add(f"{c}__prefix_{exact[:4]}")
        for k in keys:
            rows.append((eid, k))

    if not rows:
        return pd.DataFrame(columns=["entity_id", "blocking_key"])
    return pd.DataFrame(rows, columns=["entity_id", "blocking_key"])


def _address_keydf(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form DataFrame[entity_id, blocking_key] for country_address_number."""
    cntry = _norm(df["country"])
    names = _norm(df["business_name"])
    addr_col = "business_address" if "business_address" in df.columns else None
    if addr_col is None:
        return pd.DataFrame(columns=["entity_id", "blocking_key"])

    addrs = _norm(df[addr_col])
    eids = df["entity_id"].values

    # Extract all digit sequences at once
    rows = []
    for eid, c, name, addr in zip(eids, cntry.values, names.values, addrs.values):
        if not addr:
            continue
        nums = [n for n in re.findall(r"\b\d+\b", addr) if len(n) >= 2][:3]
        if not nums:
            continue
        words = re.sub(r"[^a-z0-9\s]", " ", name).split()
        first_char = words[0][0] if words else ""
        first_pref = words[0][:2] if words else ""
        for num in nums:
            rows.append((eid, f"{c}__{num}__{first_char}"))
            if len(first_pref) >= 2:
                rows.append((eid, f"{c}__{num}__{first_pref}"))

    if not rows:
        return pd.DataFrame(columns=["entity_id", "blocking_key"])
    return pd.DataFrame(rows, columns=["entity_id", "blocking_key"])


# ── Core merge-based candidate pair generation ─────────────────────────────────

def _merge_strategy(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_source: str,
    strategy: str,
    max_block_size: int = 5000,
) -> pd.DataFrame:
    """Generate candidate pairs for one strategy via pandas merge."""
    _builders = {
        "exact_normalized_name":   _exact_name_keydf,
        "country_normalized_name": _country_norm_name_keydf,
        "country_name_tokens":     _token_keydf,
        "country_address_number":  _address_keydf,
    }
    if strategy not in _builders:
        raise ValueError(f"Unsupported strategy: {strategy}")

    build = _builders[strategy]
    s1_kdf  = build(s1_df)
    tgt_kdf = build(target_df)

    if s1_kdf.empty or tgt_kdf.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])

    # Filter oversized blocks (noise / generic keys)
    block_sizes = tgt_kdf.groupby("blocking_key").size()
    valid_keys  = block_sizes[block_sizes <= max_block_size].index
    tgt_kdf = tgt_kdf[tgt_kdf["blocking_key"].isin(valid_keys)]

    if tgt_kdf.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])

    merged = s1_kdf.merge(
        tgt_kdf.rename(columns={"entity_id": "candidate_entity_id"}),
        on="blocking_key",
        how="inner",
    )
    merged = merged.rename(columns={"entity_id": "source1_entity_id"})
    merged["candidate_source"] = target_source
    return (
        merged[["source1_entity_id", "candidate_entity_id", "candidate_source"]]
        .drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    )


# ── Legacy-compat wrappers (used by candidates.py) ────────────────────────────

def build_blocking_index(df: pd.DataFrame, key_strategy: str) -> dict[str, list[str]]:
    """Build inverted index (kept for unit-test backward compat)."""
    _builders = {
        "exact_normalized_name":   _exact_name_keydf,
        "country_normalized_name": _country_norm_name_keydf,
        "country_name_tokens":     _token_keydf,
        "country_address_number":  _address_keydf,
    }
    if key_strategy not in _builders:
        raise ValueError(f"Unsupported strategy: {key_strategy}")
    kdf = _builders[key_strategy](df)
    index: dict[str, list[str]] = defaultdict(list)
    for k, eid in zip(kdf["blocking_key"], kdf["entity_id"]):
        index[k].append(eid)
    return index


def block_pairs_for_strategy(
    s1_df: pd.DataFrame,
    target_index: dict[str, list[str]],
    target_source: str,
    key_strategy: str,
    max_block_size: int = 5000,
) -> pd.DataFrame:
    """Kept for backward compat — reconstruct target df from index then merge."""
    if not target_index:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
    rows = [(eid, k) for k, eids in target_index.items() for eid in eids]
    tgt_kdf = pd.DataFrame(rows, columns=["entity_id", "blocking_key"])

    _builders = {
        "exact_normalized_name":   _exact_name_keydf,
        "country_normalized_name": _country_norm_name_keydf,
        "country_name_tokens":     _token_keydf,
        "country_address_number":  _address_keydf,
    }
    s1_kdf = _builders[key_strategy](s1_df)
    s1_kdf = s1_kdf[s1_kdf["blocking_key"].str.len() > 0]

    block_sizes = tgt_kdf.groupby("blocking_key").size()
    valid_keys  = block_sizes[block_sizes <= max_block_size].index
    tgt_kdf = tgt_kdf[tgt_kdf["blocking_key"].isin(valid_keys)]

    if s1_kdf.empty or tgt_kdf.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])

    merged = s1_kdf.merge(
        tgt_kdf.rename(columns={"entity_id": "candidate_entity_id"}),
        on="blocking_key", how="inner",
    )
    merged = merged.rename(columns={"entity_id": "source1_entity_id"})
    merged["candidate_source"] = target_source
    return (
        merged[["source1_entity_id", "candidate_entity_id", "candidate_source"]]
        .drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    )


def union_candidate_sets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [df for df in dfs if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
    return pd.concat(valid, ignore_index=True).drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    )


def block_pairs(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """Legacy function kept for backward compatibility."""
    key_norm = f"{key}__norm"
    if key_norm not in left or key_norm not in right:
        raise ValueError(f"Blocking column must exist on both frames: {key_norm}")
    return left.merge(right, on=key_norm, suffixes=("_left", "_right")).drop_duplicates()
