import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_predictions(predictions: pd.DataFrame, label_column: str = "label") -> dict[str, object]:
    if label_column not in predictions:
        raise ValueError(f"Evaluation requires column: {label_column}")
    y_true = predictions[label_column].astype(int)
    y_score = predictions["match_probability"]
    return {
        "roc_auc": roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else None,
        "report": classification_report(y_true, predictions["match"], output_dict=True),
    }
