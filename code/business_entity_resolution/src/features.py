import pandas as pd
from rapidfuzz import fuzz

def _jaccard(left: str, right: str) -> float:
    set1 = set(str(left).lower().split())
    set2 = set(str(right).lower().split())
    if not set1 and not set2:
        return 1.0
    union_len = len(set1 | set2)
    return len(set1 & set2) / union_len if union_len > 0 else 0.0

def build_features(pairs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build field-level and aggregate similarities for prefixed pair columns."""
    features = {}
    for column in columns:
        left = f"{column}_left"
        right = f"{column}_right"
        if left not in pairs or right not in pairs:
            continue
        
        # 1. Jaccard Token Similarity
        features[f"{column}_jaccard"] = [
            _jaccard(a, b) for a, b in zip(pairs[left], pairs[right])
        ]
        
        # 2. RapidFuzz String Ratio (Levenshtein distance based)
        features[f"{column}_fuzz_ratio"] = [
            fuzz.ratio(str(a).lower(), str(b).lower()) / 100.0 for a, b in zip(pairs[left], pairs[right])
        ]
        
        # 3. Length difference ratio
        features[f"{column}_len_diff"] = [
            abs(len(str(a)) - len(str(b))) / max(len(str(a)), len(str(b)), 1) for a, b in zip(pairs[left], pairs[right])
        ]

    result = pd.DataFrame(features, index=pairs.index)
    if result.empty:
        raise ValueError("No matching field columns were found in candidate pairs")
    
    # Calculate means across columns
    jaccard_cols = [c for c in result.columns if "jaccard" in c]
    fuzz_cols = [c for c in result.columns if "fuzz_ratio" in c]
    if jaccard_cols:
        result["mean_jaccard"] = result[jaccard_cols].mean(axis=1)
    if fuzz_cols:
        result["mean_fuzz"] = result[fuzz_cols].mean(axis=1)
        
    return result
