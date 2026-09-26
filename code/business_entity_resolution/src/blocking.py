from collections import defaultdict
import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES = {
    "inc", "incorporated", "ltd", "limited", "corp", "corporation",
    "llc", "pvt", "private", "co", "company", "services", "enterprises",
    "group", "store", "shop"
}

STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to",
    "co", "inc", "ltd", "corp", "llc", "pvt"
}


def normalize_string(value: object) -> str:
    """Normalize input value by converting to ASCII lowercase string."""
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def get_exact_normalized_name_key(name: object) -> str:
    """Generate exact normalized name blocking key (alphanumeric only)."""
    clean = normalize_string(name)
    return re.sub(r"[^a-z0-9]", "", clean)


def get_country_normalized_name_key(country: object, name: object) -> str:
    """Generate Country + Normalized Name blocking key (with legal suffixes removed)."""
    cntry = normalize_string(country)
    clean_name = normalize_string(name)
    words = re.sub(r"[^a-z0-9\s]", " ", clean_name).split()
    filtered = [w for w in words if w not in LEGAL_SUFFIXES]
    norm = "".join(filtered)
    if not norm:
        norm = re.sub(r"[^a-z0-9]", "", clean_name)
    return f"{cntry}__{norm}" if norm else ""


def get_country_name_token_keys(country: object, name: object) -> list[str]:
    """Generate Country + Name Token/Prefix blocking keys."""
    cntry = normalize_string(country)
    clean_name = normalize_string(name)
    exact_norm = re.sub(r"[^a-z0-9]", "", clean_name)
    words = re.sub(r"[^a-z0-9\s]", " ", clean_name).split()
    tokens = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    
    keys = set()
    # 1. Individual token 4-char prefixes
    for t in tokens[:4]:
        if len(t) >= 3:
            keys.add(f"{cntry}__{t[:4]}")
            
    # 2. Sorted pair of top tokens
    if len(tokens) >= 2:
        st = sorted(tokens[:3])
        keys.add(f"{cntry}__{st[0]}_{st[1]}")
    elif len(tokens) == 1:
        keys.add(f"{cntry}__{tokens[0]}")
        
    # 3. Exact norm name 4-char prefix
    if len(exact_norm) >= 4:
        keys.add(f"{cntry}__prefix_{exact_norm[:4]}")
        
    return list(keys)


def get_country_address_number_keys(country: object, name: object, address: object) -> list[str]:
    """Generate Country + Address Number blocking keys."""
    cntry = normalize_string(country)
    addr_clean = normalize_string(address)
    if not addr_clean:
        return []
    
    nums = re.findall(r"\b\d+\b", addr_clean)
    if not nums:
        return []
        
    clean_name = normalize_string(name)
    words = re.sub(r"[^a-z0-9\s]", " ", clean_name).split()
    first_char = words[0][0] if words and words[0] else ""
    first_prefix = words[0][:2] if words and words[0] else ""
    
    keys = set()
    for num in nums:
        if len(num) >= 2: # street number, house number, or zip code
            keys.add(f"{cntry}__{num}__{first_char}")
            if len(first_prefix) >= 2:
                keys.add(f"{cntry}__{num}__{first_prefix}")
                
    return list(keys)


def build_blocking_index(df: pd.DataFrame, key_strategy: str) -> dict[str, list[str]]:
    """Build an inverted index mapping blocking keys to target entity IDs.
    
    Args:
        df: DataFrame containing entity_id, business_name, business_address, country
        key_strategy: One of 'country_normalized_name', 'country_name_tokens',
                       'country_address_number', 'exact_normalized_name'
                       
    Returns:
        Dictionary mapping key strings to lists of entity_ids.
    """
    index = defaultdict(list)
    
    entity_ids = df["entity_id"].tolist()
    names = df["business_name"].tolist() if "business_name" in df else [None] * len(df)
    addresses = df["business_address"].tolist() if "business_address" in df else [None] * len(df)
    countries = df["country"].tolist() if "country" in df else [None] * len(df)
    
    for eid, bname, baddr, cntry in zip(entity_ids, names, addresses, countries):
        if key_strategy == "exact_normalized_name":
            key = get_exact_normalized_name_key(bname)
            if key:
                index[key].append(eid)
        elif key_strategy == "country_normalized_name":
            key = get_country_normalized_name_key(cntry, bname)
            if key:
                index[key].append(eid)
        elif key_strategy == "country_name_tokens":
            keys = get_country_name_token_keys(cntry, bname)
            for k in keys:
                if k:
                    index[k].append(eid)
        elif key_strategy == "country_address_number":
            keys = get_country_address_number_keys(cntry, bname, baddr)
            for k in keys:
                if k:
                    index[k].append(eid)
        else:
            raise ValueError(f"Unsupported blocking strategy: {key_strategy}")
            
    return index


def block_pairs_for_strategy(
    s1_df: pd.DataFrame,
    target_index: dict[str, list[str]],
    target_source: str,
    key_strategy: str,
    max_block_size: int = 5000
) -> pd.DataFrame:
    """Find candidate pairs for S1 entities against target index for a specific strategy."""
    s1_ids = s1_df["entity_id"].tolist()
    names = s1_df["business_name"].tolist() if "business_name" in s1_df else [None] * len(s1_df)
    addresses = s1_df["business_address"].tolist() if "business_address" in s1_df else [None] * len(s1_df)
    countries = s1_df["country"].tolist() if "country" in s1_df else [None] * len(s1_df)
    
    pairs = []
    for eid, bname, baddr, cntry in zip(s1_ids, names, addresses, countries):
        if key_strategy == "exact_normalized_name":
            keys = [get_exact_normalized_name_key(bname)]
        elif key_strategy == "country_normalized_name":
            keys = [get_country_normalized_name_key(cntry, bname)]
        elif key_strategy == "country_name_tokens":
            keys = get_country_name_token_keys(cntry, bname)
        elif key_strategy == "country_address_number":
            keys = get_country_address_number_keys(cntry, bname, baddr)
        else:
            keys = []
            
        seen_target_ids = set()
        for k in keys:
            if not k:
                continue
            matches = target_index.get(k, [])
            if 0 < len(matches) <= max_block_size:
                for tid in matches:
                    if tid not in seen_target_ids:
                        seen_target_ids.add(tid)
                        pairs.append((eid, tid, target_source))
                        
    return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])


def union_candidate_sets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Union multiple candidate dataframes and remove duplicate pairs."""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
    
    combined = pd.concat(valid_dfs, ignore_index=True)
    return combined.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])


def block_pairs(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """Legacy function kept for backward compatibility."""
    key_norm = f"{key}__norm"
    if key_norm not in left or key_norm not in right:
        raise ValueError(f"Blocking column must exist on both frames: {key_norm}")
    candidates = left.merge(right, on=key_norm, suffixes=("_left", "_right"))
    return candidates.drop_duplicates()

