import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support

from .candidates import evaluate_candidate_recall


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F-beta score from precision and recall.

    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    For beta=0.5, precision is weighted twice as much as recall.
    """
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta_sq) * (precision * recall) / denom


def evaluate_thresholds(
    y_true: pd.Series | np.ndarray | list,
    y_prob: pd.Series | np.ndarray | list,
    thresholds: list[float] | np.ndarray | None = None,
    beta: float = 0.5,
) -> pd.DataFrame:
    """Evaluate precision, recall, and F-beta across multiple probability thresholds.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted match probabilities.
        thresholds: Array or list of thresholds to evaluate.
        beta: Beta weight for F-score (default 0.5).

    Returns:
        DataFrame summarizing performance at each threshold.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    records = []
    for th in thresholds:
        preds = (y_p >= th).astype(int)
        tp = int(np.sum((preds == 1) & (y_t == 1)))
        fp = int(np.sum((preds == 1) & (y_t == 0)))
        fn = int(np.sum((preds == 0) & (y_t == 1)))
        tn = int(np.sum((preds == 0) & (y_t == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_val = calculate_f_beta(prec, rec, beta=beta)

        records.append({
            "threshold": round(float(th), 4),
            "precision": prec,
            "recall": rec,
            f"f_{beta:.1f}".replace(".", "_"): f_val,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return pd.DataFrame(records)


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray | list,
    y_prob: pd.Series | np.ndarray | list,
    thresholds: list[float] | np.ndarray | None = None,
    beta: float = 0.5,
) -> tuple[float, float, pd.DataFrame]:
    """Find the probability threshold maximizing the F-beta score.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities.
        thresholds: Candidate thresholds (defaults to [0.05, ..., 0.95]).
        beta: Beta weight (default 0.5).

    Returns:
        Tuple of (best_threshold, best_f_beta, evaluation_table_df)
    """
    df_eval = evaluate_thresholds(y_true, y_prob, thresholds=thresholds, beta=beta)
    f_col = f"f_{beta:.1f}".replace(".", "_")
    best_row = df_eval.loc[df_eval[f_col].idxmax()]
    best_th = float(best_row["threshold"])
    best_f = float(best_row[f_col])
    return best_th, best_f, df_eval


def evaluate_predictions(
    predictions: pd.DataFrame,
    label_column: str = "label",
    probability_column: str = "match_probability",
    prediction_column: str = "match",
    threshold: float | None = None,
    beta: float = 0.5,
) -> dict[str, object]:
    """Evaluate prediction results with ROC-AUC, classification report, and F-beta at thresholds."""
    if label_column not in predictions:
        raise ValueError(f"Evaluation requires column: {label_column}")

    y_true = predictions[label_column].astype(int)
    has_prob = probability_column in predictions

    results: dict[str, object] = {}

    if has_prob:
        y_prob = predictions[probability_column].astype(float)
        if y_true.nunique() > 1:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            results["roc_auc"] = None

        best_th, best_f_beta, th_table = find_optimal_threshold(y_true, y_prob, beta=beta)
        results["best_threshold"] = best_th
        results[f"best_f_{beta:.1f}".replace(".", "_")] = best_f_beta
        results["threshold_eval_table"] = th_table

        eval_threshold = threshold if threshold is not None else best_th
        preds = (y_prob >= eval_threshold).astype(int)
    else:
        eval_threshold = threshold
        if prediction_column in predictions:
            preds = predictions[prediction_column].astype(int)
        else:
            raise ValueError(f"Either {probability_column} or {prediction_column} must be in predictions")

    results["evaluated_threshold"] = eval_threshold
    prec, rec, f_score, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    results["precision"] = prec
    results["recall"] = rec
    results[f"f_{beta:.1f}".replace(".", "_")] = calculate_f_beta(prec, rec, beta=beta)
    results["report"] = classification_report(y_true, preds, output_dict=True, zero_division=0)

    return results


def evaluate_blocking(candidate_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict[str, float]:
    """Evaluate candidate generation blocking recall against ground truth."""
    return evaluate_candidate_recall(candidate_df, ground_truth_df)
