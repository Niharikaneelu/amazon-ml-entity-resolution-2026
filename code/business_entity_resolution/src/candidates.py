from __future__ import annotations
import pandas as pd

from .blocking import (
    block_pairs,
    block_pairs_for_strategy,
    union_candidate_sets,
)
from .normalize import normalize_columns

DEFAULT_STRATEGIES = [
    "exact_normalized_name",
    "country_normalized_name",
    "country_address_number",
]


def precompute_target_indices(target_df: pd.DataFrame, strategies: list[str]) -> dict[str, pd.DataFrame]:
    """Precompute all blocking keys for a target dataset to avoid rebuilding during chunks."""
    from .blocking import get_keys_df
    return {strat: get_keys_df(target_df, strat) for strat in strategies}

def generate_candidates_for_source(
    s1_df: pd.DataFrame,
    target_indices: dict[str, pd.DataFrame],
    target_source: str,
    strategies: list[str] | None = None,
    max_block_size: int = 5000,
) -> pd.DataFrame:
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    candidate_dfs = []
    for strat in strategies:
        strat_cands = block_pairs_for_strategy(
            s1_df=s1_df,
            target_keys_df=target_indices[strat],
            target_source=target_source,
            key_strategy=strat,
            max_block_size=max_block_size,
        )
        candidate_dfs.append(strat_cands)

    return union_candidate_sets(candidate_dfs)

def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_indices: dict[str, pd.DataFrame],
    s3_indices: dict[str, pd.DataFrame],
    strategies: list[str] | None = None,
    max_block_size: int = 5000,
) -> pd.DataFrame:
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    cands_s2 = generate_candidates_for_source(
        s1_df=s1_df,
        target_indices=s2_indices,
        target_source="source2",
        strategies=strategies,
        max_block_size=max_block_size,
    )

    cands_s3 = generate_candidates_for_source(
        s1_df=s1_df,
        target_indices=s3_indices,
        target_source="source3",
        strategies=strategies,
        max_block_size=max_block_size,
    )

    return union_candidate_sets([cands_s2, cands_s3])


def evaluate_candidate_recall(
    candidate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict[str, float]:
    """Measure candidate recall against training ground truth.

    Args:
        candidate_df: DataFrame containing source1_entity_id and candidate_entity_id.
        ground_truth_df: DataFrame containing source1_entity_id and matched_entity_ids.

    Returns:
        Dictionary of recall statistics for S2, S3, and combined overall match pairs.
    """
    s1_in_candidates = set(candidate_df["source1_entity_id"].unique()) if not candidate_df.empty else set()
    
    # Build candidate pairs sets
    candidate_pairs_s2 = set()
    candidate_pairs_s3 = set()
    candidate_pairs_all = set()

    for _, row in candidate_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        candidate_pairs_all.add((s1_id, cand_id))
        if str(cand_id).startswith("S2"):
            candidate_pairs_s2.add((s1_id, cand_id))
        elif str(cand_id).startswith("S3"):
            candidate_pairs_s3.add((s1_id, cand_id))

    # Build true ground truth pairs sets
    true_pairs_s2 = set()
    true_pairs_s3 = set()
    true_pairs_all = set()

    # Only evaluate for S1 entities present in ground truth
    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        # If restricting to candidate evaluation subset
        if s1_in_candidates and s1_id not in s1_in_candidates and len(s1_in_candidates) < len(ground_truth_df):
            continue
            
        m_str = row["matched_entity_ids"]
        if pd.isna(m_str) or not str(m_str).strip():
            continue

        for m_id in str(m_str).split(","):
            m_id = m_id.strip()
            if not m_id:
                continue
            true_pairs_all.add((s1_id, m_id))
            if m_id.startswith("S2"):
                true_pairs_s2.add((s1_id, m_id))
            elif m_id.startswith("S3"):
                true_pairs_s3.add((s1_id, m_id))

    # Compute recall
    recall_s2 = (
        len(candidate_pairs_s2 & true_pairs_s2) / len(true_pairs_s2)
        if true_pairs_s2
        else 0.0
    )
    recall_s3 = (
        len(candidate_pairs_s3 & true_pairs_s3) / len(true_pairs_s3)
        if true_pairs_s3
        else 0.0
    )
    recall_overall = (
        len(candidate_pairs_all & true_pairs_all) / len(true_pairs_all)
        if true_pairs_all
        else 0.0
    )

    num_s1_eval = len(s1_in_candidates) if s1_in_candidates else len(ground_truth_df)
    avg_candidates_per_s1 = (
        len(candidate_df) / num_s1_eval if num_s1_eval > 0 else 0.0
    )

    return {
        "candidate_recall_s2": recall_s2,
        "candidate_recall_s3": recall_s3,
        "candidate_recall_overall": recall_overall,
        "total_candidate_pairs": len(candidate_df),
        "true_pairs_s2_found": len(candidate_pairs_s2 & true_pairs_s2),
        "true_pairs_s2_total": len(true_pairs_s2),
        "true_pairs_s3_found": len(candidate_pairs_s3 & true_pairs_s3),
        "true_pairs_s3_total": len(true_pairs_s3),
        "avg_candidates_per_s1": avg_candidates_per_s1,
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

