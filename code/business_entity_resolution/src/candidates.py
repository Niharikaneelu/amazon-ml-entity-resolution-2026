import pandas as pd

from .blocking import (
    block_pairs,
    build_blocking_index,
    block_pairs_for_strategy,
    union_candidate_sets,
    _merge_strategy,
)
from .normalize import normalize_columns

DEFAULT_STRATEGIES = [
    "exact_normalized_name",
    "country_normalized_name",
    "country_name_tokens",
    "country_address_number",
]


def generate_candidates_for_source(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_source: str,
    strategies: list[str] | None = None,
    max_block_size: int = 5000,
) -> pd.DataFrame:
    """Generate candidate pairs for S1 entities matching target source records.

    Args:
        s1_df: DataFrame of Source 1 records.
        target_df: DataFrame of target records (Source 2 or Source 3).
        target_source: Source identifier string ('source2' or 'source3').
        strategies: List of blocking strategy names to apply and union.
        max_block_size: Maximum allowed block size for candidate generation.

    Returns:
        DataFrame with columns: source1_entity_id, candidate_entity_id, candidate_source
    """
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    from .blocking import (
        _exact_name_keydf, _country_norm_name_keydf,
        _token_keydf, _address_keydf,
    )
    _builders = {
        "exact_normalized_name":   _exact_name_keydf,
        "country_normalized_name": _country_norm_name_keydf,
        "country_name_tokens":     _token_keydf,
        "country_address_number":  _address_keydf,
    }

    candidate_dfs = []
    for strat in strategies:
        strat_cands = _merge_strategy(
            s1_df=s1_df,
            target_df=target_df,
            target_source=target_source,
            strategy=strat,
            max_block_size=max_block_size,
        )
        candidate_dfs.append(strat_cands)

    return union_candidate_sets(candidate_dfs)


def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    strategies: list[str] | None = None,
    max_block_size: int = 5000,
) -> pd.DataFrame:
    """Generate candidate pairs separately for S1 -> S2 and S1 -> S3 and union them.

    Args:
        s1_df: Source 1 DataFrame.
        s2_df: Source 2 DataFrame.
        s3_df: Source 3 DataFrame.
        strategies: List of blocking strategies to apply.
        max_block_size: Maximum allowed block size.

    Returns:
        Combined DataFrame matching required candidate schema:
        source1_entity_id, candidate_entity_id, candidate_source
    """
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    cands_s2 = generate_candidates_for_source(
        s1_df=s1_df,
        target_df=s2_df,
        target_source="source2",
        strategies=strategies,
        max_block_size=max_block_size,
    )

    cands_s3 = generate_candidates_for_source(
        s1_df=s1_df,
        target_df=s3_df,
        target_source="source3",
        strategies=strategies,
        max_block_size=max_block_size,
    )

    return union_candidate_sets([cands_s2, cands_s3])


def enrich_candidate_pairs(
    candidate_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich candidate pairs DataFrame with entity attributes from source DataFrames."""
    if candidate_df.empty:
        return candidate_df

    # Combine S2 and S3 for fast lookup
    target_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=["entity_id"])

    s1_sub = s1_df[["entity_id", "business_name", "business_address", "country"]].rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "s1_business_name",
            "business_address": "s1_business_address",
            "country": "s1_country",
        }
    )

    target_sub = target_df[["entity_id", "business_name", "business_address", "country"]].rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_business_name",
            "business_address": "candidate_business_address",
            "country": "candidate_country",
        }
    )

    merged = candidate_df.merge(s1_sub, on="source1_entity_id", how="left")
    enriched = merged.merge(target_sub, on="candidate_entity_id", how="left")
    return enriched


def attach_ground_truth_labels(
    candidate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach binary label column (1 = true match, 0 = candidate false match) to candidate pairs."""
    if candidate_df.empty:
        df = candidate_df.copy()
        df["label"] = []
        return df

    true_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        if pd.notna(m_str) and str(m_str).strip():
            for m_id in str(m_str).split(","):
                m_id = m_id.strip()
                if m_id:
                    true_pairs.add((s1_id, m_id))

    labels = [
        1 if (s1, cand) in true_pairs else 0
        for s1, cand in zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_id"])
    ]

    res = candidate_df.copy()
    res["label"] = labels
    return res



def evaluate_candidate_recall(
    candidate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict[str, float]:
    """Measure candidate recall against training ground truth (vectorized)."""
    if candidate_df.empty:
        return {
            "candidate_recall_s2": 0.0, "candidate_recall_s3": 0.0,
            "candidate_recall_overall": 0.0, "total_candidate_pairs": 0,
            "true_pairs_s2_found": 0, "true_pairs_s2_total": 0,
            "true_pairs_s3_found": 0, "true_pairs_s3_total": 0,
            "avg_candidates_per_s1": 0.0,
        }

    # ── Build candidate pair sets from candidate_df ──────────────────────────
    cands = candidate_df[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
    cands_s2 = cands[cands["candidate_entity_id"].str.startswith("S2")]
    cands_s3 = cands[cands["candidate_entity_id"].str.startswith("S3")]

    # ── Build true pair sets from ground_truth_df (vectorized explode) ───────
    s1_in_cands = set(cands["source1_entity_id"].unique())
    gt_filtered = ground_truth_df[ground_truth_df["source1_entity_id"].isin(s1_in_cands)].copy()
    gt_filtered = gt_filtered[gt_filtered["matched_entity_ids"].notna()]
    gt_filtered = gt_filtered[gt_filtered["matched_entity_ids"].str.strip() != ""]

    if gt_filtered.empty:
        true_pairs = pd.DataFrame(columns=["source1_entity_id", "matched_entity_id"])
    else:
        gt_filtered = gt_filtered.copy()
        gt_filtered["matched_entity_id"] = gt_filtered["matched_entity_ids"].str.split(",")
        gt_exploded = gt_filtered.explode("matched_entity_id")
        gt_exploded["matched_entity_id"] = gt_exploded["matched_entity_id"].str.strip()
        true_pairs = gt_exploded[gt_exploded["matched_entity_id"] != ""][
            ["source1_entity_id", "matched_entity_id"]
        ].drop_duplicates()

    true_s2 = true_pairs[true_pairs["matched_entity_id"].str.startswith("S2")]
    true_s3 = true_pairs[true_pairs["matched_entity_id"].str.startswith("S3")]

    # ── Compute recall via merge ──────────────────────────────────────────────
    def _recall(cand: pd.DataFrame, truth: pd.DataFrame, cand_col: str, truth_col: str) -> tuple[int, int]:
        if truth.empty:
            return 0, 0
        found = cand.merge(truth, left_on=["source1_entity_id", cand_col],
                           right_on=["source1_entity_id", truth_col])
        return len(found), len(truth)

    found_s2, total_s2 = _recall(cands_s2, true_s2, "candidate_entity_id", "matched_entity_id")
    found_s3, total_s3 = _recall(cands_s3, true_s3, "candidate_entity_id", "matched_entity_id")
    found_all = found_s2 + found_s3
    total_all  = total_s2 + total_s3

    num_s1 = len(s1_in_cands) if s1_in_cands else len(ground_truth_df)
    return {
        "candidate_recall_s2":      found_s2 / total_s2 if total_s2 else 0.0,
        "candidate_recall_s3":      found_s3 / total_s3 if total_s3 else 0.0,
        "candidate_recall_overall": found_all / total_all if total_all else 0.0,
        "total_candidate_pairs":    len(candidate_df),
        "true_pairs_s2_found":      found_s2,
        "true_pairs_s2_total":      total_s2,
        "true_pairs_s3_found":      found_s3,
        "true_pairs_s3_total":      total_s3,
        "avg_candidates_per_s1":    len(candidate_df) / num_s1 if num_s1 else 0.0,
    }



def format_candidates_for_submission(
    candidate_df: pd.DataFrame,
    s1_entity_ids: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Format candidate pair records into submission schema candidate_pairs.tsv.

    Columns: source1_entity_id, candidate_entity_ids (comma-separated string).
    """
    if candidate_df.empty:
        grouped = {}
    else:
        grouped = candidate_df.groupby("source1_entity_id")["candidate_entity_id"].apply(
            lambda ids: ",".join(dict.fromkeys(ids))
        ).to_dict()

    if s1_entity_ids is None:
        s1_entity_ids = list(grouped.keys())

    rows = []
    for s1_id in s1_entity_ids:
        cands = grouped.get(s1_id, "")
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands})

    return pd.DataFrame(rows)


def generate_candidates(
    left: pd.DataFrame, right: pd.DataFrame, block_column: str = "business_name"
) -> pd.DataFrame:
    """Legacy function kept for backward compatibility."""
    left_norm = normalize_columns(left, [block_column])
    right_norm = normalize_columns(right, [block_column])
    return block_pairs(left_norm, right_norm, block_column)

