"""Run the baseline business entity resolution pipeline with chunking."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution"))

from src.candidates import generate_all_candidates, format_candidates_for_submission, precompute_target_indices, DEFAULT_STRATEGIES
from src.blocking import get_keys_df
from src.train import train_model
from src.predict import predict_pairs

def process_chunk(s1_chunk, s2_indices, s3_indices, s2_df, s3_df, model_path, is_first_chunk, output_dir):
    candidates = generate_all_candidates(s1_chunk, s2_indices, s3_indices)
    
    if candidates.empty:
        cands_out = pd.DataFrame({"source1_entity_id": s1_chunk["entity_id"].unique(), "candidate_entity_ids": ""})
        matches_out = pd.DataFrame({"source1_entity_id": s1_chunk["entity_id"].unique(), "matched_entity_ids": ""})
    else:
        s23_train = pd.concat([s2_df, s3_df], ignore_index=True)
        pairs = candidates.merge(
            s1_chunk[["entity_id", "business_name", "business_address"]], 
            left_on="source1_entity_id", right_on="entity_id", how="left"
        ).rename(columns={"business_name": "business_name_left", "business_address": "business_address_left"})
        
        pairs = pairs.merge(
            s23_train[["entity_id", "business_name", "business_address"]], 
            left_on="candidate_entity_id", right_on="entity_id", how="left"
        ).rename(columns={"business_name": "business_name_right", "business_address": "business_address_right"})
        
        predictions = predict_pairs(pairs, model_path, threshold=0.7) 
        
        candidate_out = format_candidates_for_submission(candidates, s1_chunk["entity_id"].unique())
        
        positive_preds = predictions[predictions["match"] == 1]
        matches_out = format_candidates_for_submission(positive_preds, s1_chunk["entity_id"].unique())
        matches_out = matches_out.rename(columns={"candidate_entity_ids": "matched_entity_ids"})

    mode = "w" if is_first_chunk else "a"
    header = is_first_chunk
    
    candidate_out.to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False, mode=mode, header=header)
    matches_out.to_csv(output_dir / "matching_results.tsv", sep="\t", index=False, mode=mode, header=header)

def main() -> None:
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "student_resource" / "dataset"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Target Reference Data (S2 & S3) entirely into memory...")
    s2_train = pd.read_csv(dataset_dir / "train" / "train_source2.tsv", sep="\t")
    s3_train = pd.read_csv(dataset_dir / "train" / "train_source3.tsv", sep="\t")
    ground_truth = pd.read_csv(dataset_dir / "train" / "train_ground_truth.tsv", sep="\t")
    
    # --- 1. TRAINING PHASE ---
    print("\n--- PHASE 1: Training the XGBoost Model ---")
    s1_train_sample = pd.read_csv(dataset_dir / "train" / "train_source1.tsv", sep="\t", nrows=10000)
    print(f"Generating training candidates from {len(s1_train_sample)} S1 records...")
    
    # Generate training candidates by building temporary indices just for training
    train_cands_list = []
    for strat in DEFAULT_STRATEGIES:
        s2_idx_temp = {strat: get_keys_df(s2_train, strat)}
        s3_idx_temp = {strat: get_keys_df(s3_train, strat)}
        train_cands_list.append(generate_all_candidates(s1_train_sample, s2_idx_temp, s3_idx_temp, [strat]))
    train_cands = pd.concat(train_cands_list, ignore_index=True).drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    
    gt_pairs = set()
    for _, row in ground_truth.iterrows():
        s1 = row["source1_entity_id"]
        matches = str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"]) else []
        for m in matches:
            if m.strip():
                gt_pairs.add((s1, m.strip()))
                
    gt_df = pd.DataFrame(list(gt_pairs), columns=["source1_entity_id", "candidate_entity_id"])
    gt_df["label"] = 1
    train_cands = train_cands.merge(gt_df, on=["source1_entity_id", "candidate_entity_id"], how="left")
    train_cands["label"] = train_cands["label"].fillna(0).astype(int)
    
    s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
    train_pairs = train_cands.merge(
        s1_train_sample[["entity_id", "business_name", "business_address"]], left_on="source1_entity_id", right_on="entity_id", how="left"
    ).rename(columns={"business_name": "business_name_left", "business_address": "business_address_left"})
    
    train_pairs = train_pairs.merge(
        s23_train[["entity_id", "business_name", "business_address"]], left_on="candidate_entity_id", right_on="entity_id", how="left"
    ).rename(columns={"business_name": "business_name_right", "business_address": "business_address_right"})
    
    model_path = output_dir / "model.joblib"
    print("Training XGBoost...")
    train_model(train_pairs, ["business_name", "business_address"], "label", model_path)
    
    # --- 2. INFERENCE PHASE (Strategy-by-Strategy Chunking) ---
    print("\n--- PHASE 2: Inference & Output Generation ---")
    chunk_size = 50000
    
    # We write all raw predictions to a temp file, then deduplicate at the very end
    temp_preds_path = output_dir / "temp_predictions.tsv"
    is_first_write = True
    
    for strat in DEFAULT_STRATEGIES:
        print(f"\n>> Processing Strategy: {strat}")
        print("  Precomputing S2 & S3 index for this strategy...")
        s2_idx = {strat: get_keys_df(s2_train, strat)}
        s3_idx = {strat: get_keys_df(s3_train, strat)}
        
        chunk_iter = pd.read_csv(dataset_dir / "train" / "train_source1.tsv", sep="\t", chunksize=chunk_size)
        for i, s1_chunk in enumerate(chunk_iter):
            print(f"  Scoring chunk {i+1}...")
            cands = generate_all_candidates(s1_chunk, s2_idx, s3_idx, [strat])
            
            if not cands.empty:
                pairs = cands.merge(
                    s1_chunk[["entity_id", "business_name", "business_address"]], 
                    left_on="source1_entity_id", right_on="entity_id", how="left"
                ).rename(columns={"business_name": "business_name_left", "business_address": "business_address_left"})
                
                pairs = pairs.merge(
                    s23_train[["entity_id", "business_name", "business_address"]], 
                    left_on="candidate_entity_id", right_on="entity_id", how="left"
                ).rename(columns={"business_name": "business_name_right", "business_address": "business_address_right"})
                
                preds = predict_pairs(pairs, model_path, threshold=0.7)
                preds = preds[["source1_entity_id", "candidate_entity_id", "match"]]
                
                mode = "w" if is_first_write else "a"
                preds.to_csv(temp_preds_path, sep="\t", index=False, mode=mode, header=is_first_write)
                is_first_write = False
                
    print("\n--- PHASE 3: Deduplicating and Formatting Output ---")
    print("Reading all predictions...")
    all_preds = pd.read_csv(temp_preds_path, sep="\t")
    # A pair might be generated by multiple strategies, take the max match score (1 if matched by any)
    final_preds = all_preds.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)["match"].max()
    
    print("Formatting final submissions...")
    # All candidates
    s1_full = pd.read_csv(dataset_dir / "train" / "train_source1.tsv", sep="\t", usecols=["entity_id"])
    s1_ids = s1_full["entity_id"].unique()
    
    cand_out = format_candidates_for_submission(final_preds, s1_ids)
    cand_out.to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    
    # Matches only
    positives = final_preds[final_preds["match"] == 1]
    match_out = format_candidates_for_submission(positives, s1_ids)
    match_out = match_out.rename(columns={"candidate_entity_ids": "matched_entity_ids"})
    match_out.to_csv(output_dir / "matching_results.tsv", sep="\t", index=False)
    
    if temp_preds_path.exists():
        temp_preds_path.unlink()
        
    print("\nDone! Entire dataset processed out-of-core.")

if __name__ == "__main__":
    main()
