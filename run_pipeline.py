"""Run the baseline business entity resolution pipeline."""
import sys
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution"))

from src.candidates import generate_all_candidates, evaluate_candidate_recall, format_candidates_for_submission
from src.features import build_features
from src.train import train_model
from src.predict import predict_pairs
from src.evaluate import evaluate_predictions

def main() -> None:
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "student_resource" / "dataset"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading training data...")
    # For quick testing, we can limit the number of rows read. Set to None to read everything.
    # Since the dataset is huge, we'll read a small chunk just to prove the pipeline works end-to-end.
    s1_train = pd.read_csv(dataset_dir / "train" / "train_source1.tsv", sep="\t")
    s2_train = pd.read_csv(dataset_dir / "train" / "train_source2.tsv", sep="\t")
    s3_train = pd.read_csv(dataset_dir / "train" / "train_source3.tsv", sep="\t")
    ground_truth = pd.read_csv(dataset_dir / "train" / "train_ground_truth.tsv", sep="\t")
    
    print(f"Generating candidate pairs for {len(s1_train)} Source 1 records...")
    candidates = generate_all_candidates(s1_train, s2_train, s3_train)
    
    print(f"Generated {len(candidates)} candidate pairs.")
    
    print("Evaluating blocking recall...")
    recall_stats = evaluate_candidate_recall(candidates, ground_truth)
    for k, v in recall_stats.items():
        print(f"  {k}: {v}")
        
    print("Labeling candidates with ground truth...")
    # Expand ground truth into pairs
    gt_pairs = set()
    for _, row in ground_truth.iterrows():
        s1 = row["source1_entity_id"]
        matches = str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"]) else []
        for m in matches:
            if m.strip():
                gt_pairs.add((s1, m.strip()))
                
    candidates["label"] = candidates.apply(
        lambda x: 1 if (x["source1_entity_id"], x["candidate_entity_id"]) in gt_pairs else 0, 
        axis=1
    )
    
    print("Merging text features for ML model...")
    s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    train_pairs = candidates.merge(
        s1_train[["entity_id", "business_name", "business_address"]], 
        left_on="source1_entity_id", 
        right_on="entity_id", 
        how="left"
    ).rename(columns={"business_name": "business_name_left", "business_address": "business_address_left"})
    
    train_pairs = train_pairs.merge(
        s23_train[["entity_id", "business_name", "business_address"]], 
        left_on="candidate_entity_id", 
        right_on="entity_id", 
        how="left"
    ).rename(columns={"business_name": "business_name_right", "business_address": "business_address_right"})
    
    # ML model expects features based on these fields
    fields = ["business_name", "business_address"]
    model_path = output_dir / "model.joblib"
    
    print("Training model...")
    # The training can be slow due to difflib.SequenceMatcher in features.py
    train_model(train_pairs, fields, "label", model_path)
    
    print("Predicting on training set (for evaluation)...")
    predictions = predict_pairs(train_pairs, model_path, threshold=0.5)
    metrics = evaluate_predictions(predictions, "label")
    print(f"Training ROC AUC: {metrics['roc_auc']}")
    
    # Generate final matches format for candidate pairs
    print("Writing candidates to output/candidate_pairs.tsv...")
    candidate_out = format_candidates_for_submission(candidates, s1_train["entity_id"].unique())
    candidate_out.to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    
    # Generate final matches format for matching results
    print("Writing positive predictions to output/matching_results.tsv...")
    positive_preds = predictions[predictions["match"] == 1]
    matches_out = format_candidates_for_submission(positive_preds, s1_train["entity_id"].unique())
    # Rename column to match required output
    matches_out = matches_out.rename(columns={"candidate_entity_ids": "matched_entity_ids"})
    matches_out.to_csv(output_dir / "matching_results.tsv", sep="\t", index=False)
    
    print("Done! You can run 'python student_resource/utils/validate_submission.py ...' to verify outputs.")

if __name__ == "__main__":
    main()
