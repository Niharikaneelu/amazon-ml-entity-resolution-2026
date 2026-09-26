# Amazon ML Entity Resolution 2026

A reproducible baseline for matching noisy business records to the same underlying entity.

## Layout

- `code/business_entity_resolution/src`: reusable pipeline modules
- `dataset`: input data (ignored by git)
- `output`: models, predictions, and metrics (ignored by git)
- `notebooks`: exploratory noise and error analysis

## Input contract

Place `dataset/train.csv` in the repository. It must contain `label` (`0` or `1`) and paired columns using these names when available:

`name_left`, `name_right`, `address_left`, `address_right`, `city_left`, `city_right`, `postal_code_left`, `postal_code_right`, `phone_left`, `phone_right`, `email_left`, `email_right`.

At least one paired field is required. The pipeline automatically uses the paired fields that are present.

## Run

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The trained model is written to `output/model.joblib` and scored pairs to `output/predictions.csv`. Set `PipelineConfig.threshold` to tune the precision/recall tradeoff.
