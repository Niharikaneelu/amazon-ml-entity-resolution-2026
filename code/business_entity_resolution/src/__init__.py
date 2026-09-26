"""Business entity resolution package."""

from .blocking import (
    get_keys_df,
    block_pairs_for_strategy,
    union_candidate_sets,
)
from .candidates import (
    generate_candidates_for_source,
    generate_all_candidates,
    evaluate_candidate_recall,
    format_candidates_for_submission,
    precompute_target_indices,
)

__all__ = [
    "get_keys_df",
    "precompute_target_indices",
    "block_pairs_for_strategy",
    "union_candidate_sets",
    "generate_candidates_for_source",
    "generate_all_candidates",
    "evaluate_candidate_recall",
    "format_candidates_for_submission",
]
