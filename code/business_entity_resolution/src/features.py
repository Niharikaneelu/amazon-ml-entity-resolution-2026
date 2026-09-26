import pandas as pd

def _similarity(left: object, right: object) -> float:
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
        features[f"{column}_similarity"] = [
            _similarity(a, b) for a, b in zip(pairs[left], pairs[right])
        ]
    result = pd.DataFrame(features, index=pairs.index)
    if result.empty:
        raise ValueError("No matching field columns were found in candidate pairs")
    result["mean_similarity"] = result.mean(axis=1)
    return result
