"""Run the baseline business entity resolution pipeline."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution"))

from src.config import PipelineConfig
from src.evaluate import evaluate_predictions
from src.load_data import read_table
from src.predict import predict_pairs
from src.train import train_model


FIELDS = ["name", "address", "city", "postal_code", "phone", "email"]


def main() -> None:
    config = PipelineConfig()
    train = read_table(config.train_path)
    fields = [field for field in FIELDS if f"{field}_left" in train and f"{field}_right" in train]
    if not fields:
        raise ValueError("train.csv must contain paired columns such as name_left and name_right")
    if "label" not in train:
        raise ValueError("train.csv must contain a binary label column")

    train_model(train, fields, "label", config.model_path)
    predictions = predict_pairs(train, config.model_path, config.threshold)
    predictions.to_csv(config.predictions_path, index=False)
    metrics = evaluate_predictions(predictions)
    print(f"ROC AUC: {metrics['roc_auc']}")
    print(f"Wrote {len(predictions)} predictions to {config.predictions_path}")


if __name__ == "__main__":
    main()
