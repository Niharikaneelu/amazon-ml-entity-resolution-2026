import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from .candidates import evaluate_candidate_recall


def compute_f05_score(precision: float, recall: float) -> float:
    """Compute F_0.5 score given precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def compute_macro_f05(
    matching_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> dict[str, float]:
    """Compute macro-averaged F_0.5 score per Source 1 entity.

    Rules from problem statement:
    - Singletons: S1 entity with no true matches. Scores 1.0 when empty list predicted,
      and 0.0 when any match predicted.
    - Non-singletons: Precision, Recall, and F_0.5 calculated for predicted match set vs true set.
    - Final score: Macro average across all S1 entities in ground_truth_df.
    """
    # Build ground truth dict
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = row["matched_entity_ids"]
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = {m.strip() for m in str(raw_matches).split(",") if m.strip()}

    # Build predicted map
    pred_map = {}
    for _, row in matching_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = row["matched_entity_ids"]
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            pred_map[s1_id] = set()
        else:
            pred_map[s1_id] = {m.strip() for m in str(raw_matches).split(",") if m.strip()}

    scores = []
    precisions = []
    recalls = []
    singleton_count = 0
    singleton_correct = 0

    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())

        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
        else:
            if len(pred_set) == 0:
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                f05 = compute_f05_score(p, r)
                scores.append(f05)
                precisions.append(p)
                recalls.append(r)

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_accuracy": singleton_acc,
        "total_eval_entities": len(gt_map),
        "total_singletons": singleton_count,
        "correct_singletons": singleton_correct,
    }


def evaluate_predictions(
    predictions: pd.DataFrame, label_column: str = "label"
) -> dict[str, object]:
    if label_column not in predictions:
        raise ValueError(f"Evaluation requires column: {label_column}")
    y_true = predictions[label_column].astype(int)
    y_score = predictions["match_probability"]
    return {
        "roc_auc": roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else None,
        "report": classification_report(
            y_true, predictions["match"], output_dict=True
        ),
    }


def evaluate_blocking(
    candidate_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> dict[str, float]:
    """Evaluate candidate generation blocking recall against ground truth."""
    return evaluate_candidate_recall(candidate_df, ground_truth_df)


