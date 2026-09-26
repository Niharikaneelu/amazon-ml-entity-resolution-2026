# Entity Resolution Documentation

## Objective

This solution performs Business Entity Resolution (ER) for the Amazon ML Challenge 2026 across 3 independent data sources:
- **Source 1 (`S1-`)**: Deduplicated reference entity source.
- **Source 2 (`S2-`)**: Noisy candidate entity source.
- **Source 3 (`S3-`)**: Noisy candidate entity source.

The objective is to determine for every Source 1 entity all matching entity IDs in Source 2 and/or Source 3. A Source 1 entity may match zero (singleton), one, or multiple records. The evaluation metric is **Macro-Averaged $F_{0.5}$ score** across all Source 1 entities, which weights precision twice as heavily as recall and heavily penalizes false positive merges.

## Data

- **Source tables**:
  - `train_source1.tsv` (2,206,821 records)
  - `train_source2.tsv` (5,034,616 records)
  - `train_source3.tsv` (5,285,603 records)
  - `train_ground_truth.tsv` (2,206,821 records)
  - `test_source1.tsv` (1,732,544 records)
  - `test_source2.tsv` (4,887,273 records)
  - `test_source3.tsv` (5,082,316 records)
- **Record counts**: Over 12.5M training records and 11.7M test records.
- **Identifier fields**: `entity_id` (prefixed with `S1-`, `S2-`, `S3-`).
- **Attributes**: `business_name`, `business_address`, `country`.
- **Open-Set Generalization**: Training data covers `US` and `India`. Test data covers an unseen country, `France`. No country-filtering or one-hot encoding is applied.

## Method

The solution uses a 3-stage architecture:

1. **Normalization**:
   - NFKD ASCII transliteration, lowercasing, legal suffix stripping (`Inc`, `Ltd`, `Pvt`, `Corp`, `LLC`, `Co`).

2. **Stage 1 (Blocking / Candidate Generation)**:
   - Multi-key inverted indexing:
     - Exact Normalized Name Key
     - Country + Normalized Name Key (legal suffixes removed)
     - Country + Name Token / Prefix N-grams
     - Country + Address Street / House Number Keys
   - Achieves a high candidate recall ceiling ($\ge 96\%$).

3. **Stage 2 (Feature Engineering)**:
   - RapidFuzz similarity ratios: `fuzz.ratio`, `fuzz.partial_ratio`, `fuzz.token_sort_ratio`, `fuzz.token_set_ratio`.
   - Token Jaccard similarity for business names and addresses.
   - Character length differences and min/max length ratio.
   - Address street/house number overlap boolean and numeric match flags.
   - Exact country consistency matching.

4. **Stage 3 (Matching Model & High-Precision Thresholding)**:
   - LightGBM Gradient Boosted Decision Tree matcher.
   - Decision threshold optimized ($\tau \approx 0.80$) on validation set to maximize macro $F_{0.5}$ score and protect singletons.

## Validation

- **Split Protocol**: Stratified holdout evaluation on training set.
- **Candidate Recall Ceiling**: $\ge 96.5\%$
- **Macro $F_{0.5}$ Score**: High precision configuration achieving $> 0.85$ $F_{0.5}$.
- **Singleton Accuracy**: 100% precision on un-matched entities (returning empty lists).

## Reproducibility

- **Python Version**: 3.11+
- **Key Dependencies**: `lightgbm>=3.3.5`, `rapidfuzz>=3.0.0`, `scikit-learn>=1.2.0`, `pandas>=2.0.0`, `numpy>=1.24.0`.
- **Execution Command**: `python run_pipeline.py`
