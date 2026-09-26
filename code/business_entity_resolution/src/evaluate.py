import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from .candidates import evaluate_candidate_recall


def evaluate_predictions(predictions: pd.DataFrame, label_column: str = "label") -> dict[str, object]:
    if label_column not in predictions:
        raise ValueError(f"Evaluation requires column: {label_column}")
    y_true = predictions[label_column].astype(int)
    y_score = predictions["match_probability"]
    return {
        "roc_auc": roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else None,
        "report": classification_report(y_true, predictions["match"], output_dict=True),
    }


def evaluate_blocking(candidate_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict[str, float]:
    """Evaluate candidate generation blocking recall against ground truth."""
    return evaluate_candidate_recall(candidate_df, ground_truth_df)

