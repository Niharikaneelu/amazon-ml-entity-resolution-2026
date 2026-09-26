"""Business entity resolution package."""

from .blocking import (
    build_blocking_index,
    block_pairs_for_strategy,
    union_candidate_sets,
)
from .candidates import (
    generate_candidates_for_source,
    generate_all_candidates,
    evaluate_candidate_recall,
    format_candidates_for_submission,
)

__all__ = [
    "build_blocking_index",
    "block_pairs_for_strategy",
    "union_candidate_sets",
    "generate_candidates_for_source",
    "generate_all_candidates",
    "evaluate_candidate_recall",
    "format_candidates_for_submission",
]

