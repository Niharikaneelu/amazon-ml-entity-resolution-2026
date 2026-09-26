import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def _get_tokens(text: object) -> set[str]:
    if pd.isna(text) or text is None:
        return set()
    return set(re.findall(r"\b\w{2,}\b", str(text).lower()))


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _address_number_match(addr_a: object, addr_b: object) -> float:
    if pd.isna(addr_a) or pd.isna(addr_b) or not addr_a or not addr_b:
        return 0.5
    nums_a = set(re.findall(r"\b\d+\b", str(addr_a)))
    nums_b = set(re.findall(r"\b\d+\b", str(addr_b)))
    if not nums_a and not nums_b:
        return 0.5
    if not nums_a or not nums_b:
        return 0.25
    overlap = len(nums_a & nums_b)
    if overlap > 0:
        return 1.0
    return 0.0


def extract_pair_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
    country_a: str,
    country_b: str
) -> dict[str, float]:
    """Extract fine-grained similarity features for a single entity pair."""
    na = str(name_a) if pd.notna(name_a) else ""
    nb = str(name_b) if pd.notna(name_b) else ""
    ad_a = str(addr_a) if pd.notna(addr_a) else ""
    ad_b = str(addr_b) if pd.notna(addr_b) else ""
    ca = str(country_a).strip().lower() if pd.notna(country_a) else ""
    cb = str(country_b).strip().lower() if pd.notna(country_b) else ""

    tokens_na = _get_tokens(na)
    tokens_nb = _get_tokens(nb)
    tokens_ada = _get_tokens(ad_a)
    tokens_adb = _get_tokens(ad_b)

    len_a = len(na)
    len_b = len(nb)
    max_len = max(len_a, len_b, 1)
    min_len = min(len_a, len_b)

    return {
        "name_ratio": fuzz.ratio(na, nb) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(na, nb) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(na, nb) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(na, nb) / 100.0,
        "name_jaccard": _jaccard_similarity(tokens_na, tokens_nb),
        "name_len_diff": float(abs(len_a - len_b)),
        "name_len_ratio": float(min_len / max_len),
        "address_ratio": fuzz.ratio(ad_a, ad_b) / 100.0 if ad_a and ad_b else 0.0,
        "address_token_set_ratio": fuzz.token_set_ratio(ad_a, ad_b) / 100.0 if ad_a and ad_b else 0.0,
        "address_jaccard": _jaccard_similarity(tokens_ada, tokens_adb),
        "address_number_match": _address_number_match(ad_a, ad_b),
        "country_match": 1.0 if ca == cb and ca != "" else (0.5 if not ca or not cb else 0.0),
    }


def build_features(pairs: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Build similarity features for candidate pairs DataFrame.

    Supports candidate pair frames with (s1_business_name, candidate_business_name)
    or legacy paired columns (business_name_left, business_name_right).
    """
    if pairs.empty:
        return pd.DataFrame()

    # Determine column mapping
    if "s1_business_name" in pairs.columns and "candidate_business_name" in pairs.columns:
        name_l, name_r = "s1_business_name", "candidate_business_name"
        addr_l, addr_r = "s1_business_address", "candidate_business_address"
        cntry_l, cntry_r = "s1_country", "candidate_country"
    elif "business_name_left" in pairs.columns and "business_name_right" in pairs.columns:
        name_l, name_r = "business_name_left", "business_name_right"
        addr_l, addr_r = "business_address_left", "business_address_right"
        cntry_l, cntry_r = "country_left", "country_right"
    else:
        name_l, name_r = "name_left", "name_right"
        addr_l, addr_r = "address_left", "address_right"
        cntry_l, cntry_r = "country_left", "country_right"

    names_a = pairs[name_l].tolist() if name_l in pairs else [""] * len(pairs)
    names_b = pairs[name_r].tolist() if name_r in pairs else [""] * len(pairs)
    addrs_a = pairs[addr_l].tolist() if addr_l in pairs else [""] * len(pairs)
    addrs_b = pairs[addr_r].tolist() if addr_r in pairs else [""] * len(pairs)
    cntrys_a = pairs[cntry_l].tolist() if cntry_l in pairs else [""] * len(pairs)
    cntrys_b = pairs[cntry_r].tolist() if cntry_r in pairs else [""] * len(pairs)

    feature_rows = [
        extract_pair_features(na, nb, ada, adb, ca, cb)
        for na, nb, ada, adb, ca, cb in zip(
            names_a, names_b, addrs_a, addrs_b, cntrys_a, cntrys_b
        )
    ]

    res = pd.DataFrame(feature_rows, index=pairs.index)
    res["mean_similarity"] = res[["name_token_set_ratio", "address_token_set_ratio"]].mean(axis=1)
    return res

