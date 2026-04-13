# API Reference

Base URL (local): `http://localhost:7860`

## GET /health

Health and model-load status.

```json
{
  "status": "ok",
  "models_loaded": {
    "random_forest": true,
    "neural_network": true
  }
}
```

## POST /predict

Predict aqueous solubility for a SMILES string.

### Request

```json
{
  "smiles": "CCO"
}
```

### Response (valid)

```json
{
  "smiles": "CCO",
  "valid": true,
  "predictions": {
    "random_forest": -0.7312,
    "neural_network": -0.5841
  },
  "descriptors": {
    "molecular_weight": 46.07,
    "logp": -0.0014,
    "hbd": 1,
    "hba": 1,
    "tpsa": 20.23
  },
  "molecule_name": "Ethanol"
}
```

### Response (invalid)

```json
{
  "smiles": "NOT_VALID",
  "valid": false,
  "error": "Could not parse SMILES string"
}
```

Notes:

- values are `log(solubility)` in mol/L
- more negative means less soluble
- successful predictions are persisted to history on a best-effort basis

## GET /examples

Returns example molecules with predictions and descriptors. Useful for pre-populating the dashboard.

Response shape: array of objects containing:

- `smiles`
- `name`
- `predictions`
- `descriptors`

## GET /history

Paginated prediction history.

### Query params

- `limit` — default `50`, min `1`, max `200`
- `offset` — default `0`
- `smiles` — optional exact-match filter

### Response

```json
{
  "items": [
    {
      "id": 1,
      "smiles": "CCO",
      "molecule_name": "Ethanol",
      "rf_prediction": -0.1,
      "nn_prediction": -0.2,
      "descriptors": {
        "molecular_weight": 46.07
      },
      "created_at": "2026-04-12T12:34:56+00:00",
      "rf_model_version": "2026.04.12-101500-rf",
      "nn_model_version": "2026.04.12-101500-nn"
    }
  ],
  "total": 1
}
```

Rows are ordered newest-first.

## GET /models

Returns recent model-version records, with active versions first.

### Response

```json
[
  {
    "id": 7,
    "name": "random_forest",
    "version": "2026.04.12-101500-rf",
    "mlflow_run_id": "abc123...",
    "artifact_path": "/abs/path/models/random_forest.pkl",
    "trained_at": "2026-04-12T10:15:00+00:00",
    "cv_r2_mean": 0.71,
    "cv_rmse_mean": 1.17,
    "test_r2": 0.70,
    "test_rmse": 1.20,
    "hyperparameters": {
      "n_estimators": 200
    },
    "is_active": true
  }
]
```

## Local Runtime Notes

- `uvicorn api.main:app --port 7860` starts the API
- startup runs `alembic upgrade head` unless `SOLPREDICT_SKIP_MIGRATIONS=1`
- if active model versions exist in the DB, the API loads those artifact paths first
- otherwise it falls back to `models/random_forest.pkl` and `models/neural_network.pt`
