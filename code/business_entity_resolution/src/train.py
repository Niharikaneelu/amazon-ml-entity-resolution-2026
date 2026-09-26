from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from .features import build_features

def train_model(pairs: pd.DataFrame, fields: list[str], label_column: str, model_path: Path):
    features = build_features(pairs, fields)
    
    # Calculate scale_pos_weight to handle class imbalance (acts like class_weight='balanced')
    pos_count = pairs[label_column].sum()
    neg_count = len(pairs) - pos_count
    scale_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(features, pairs[label_column].astype(int))
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "fields": fields}, model_path)
    return model
