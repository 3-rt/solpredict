# Architecture

SolPredict now has four connected layers: training, shared application code, a FastAPI serving layer, and a single-page frontend.

```text
                       ┌──────────────────────────┐
                       │      scripts/train.py    │
                       │  thin CLI + migrations   │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │      solpredict/training/       │
                    │ load -> cache -> split -> tune  │
                    │ -> retrain -> export -> register│
                    └────────────┬────────────────────┘
                                 │
          ┌──────────────────────┼────────────────────────┐
          ▼                      ▼                        ▼
 ┌─────────────────┐   ┌──────────────────┐    ┌────────────────────┐
 │ models/*.pkl/pt │   │ data/results.json│    │ model_versions +   │
 │ local artifacts │   │ dashboard metrics│    │ predictions tables │
 └────────┬────────┘   └──────────────────┘    └─────────┬──────────┘
          │                                              │
          ▼                                              ▼
   ┌────────────────────────────────────────────────────────────┐
   │                      FastAPI API                           │
   │ /predict  /health  /examples  /history  /models           │
   │ startup migrations + active model loading                 │
   └──────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
                 ┌──────────────────────────────┐
                 │ Next.js single-page dashboard│
                 │ predict + history + compare  │
                 └──────────────────────────────┘
```

## Shared Package

The `solpredict/` package is the system core.

- `featurize.py`
  - converts SMILES into 2048-bit Morgan fingerprints
  - computes descriptor payloads used by the API/UI
- `model.py`
  - defines `SolubilityMLP`
  - now accepts configurable hidden dimensions and dropout so tuned models can be reconstructed at inference time
- `config.py`
  - centralizes environment-backed settings for DB, MLflow, paths, and training parameters
- `db/`
  - SQLAlchemy models, engine/session creation, and repository helpers
- `training/`
  - modular training implementation introduced in Phase 2
- `tracking.py`
  - MLflow tracking setup and model logging wrappers

## Training Layer

The training flow is intentionally split into focused modules:

- `training/data.py`
  - loads ESOL
  - normalizes dataset columns
  - caches featurized fingerprints/descriptors under `data/cache/`
- `training/cv.py`
  - shared metric computation and K-fold aggregation
- `training/tune.py`
  - Optuna search for RF and MLP hyperparameters
  - supports `--skip-tune` via shared default configs
- `training/pipeline.py`
  - orchestrates holdout split, tuning, retraining, artifact export, MLflow logging, and DB registration

`scripts/train.py` is now a thin CLI wrapper. It runs Alembic migrations first, opens a DB session, and delegates the real work to `run_training_pipeline(...)`.

## Persistence Layer

Phase 1 introduced a small application database used by both training and serving.

- `model_versions`
  - stores registered RF/NN artifacts
  - includes active flag, hyperparameters, CV/test metrics, trained timestamp, and MLflow run id
- `predictions`
  - stores `/predict` history rows
  - links back to active model versions when available
  - persists descriptors, timestamp, client IP, and known molecule name

Alembic owns schema evolution through `alembic/versions/`.

## API Layer

The FastAPI app lives under `api/`.

- `api/main.py`
  - creates the app
  - runs `alembic upgrade head` at startup unless disabled
  - loads active model versions from the DB
  - reconstructs the tuned MLP architecture from stored hyperparameters before loading weights
- `api/routes/predict.py`
  - prediction and example routes
  - `/predict` writes best-effort history rows without changing the prediction contract
- `api/routes/history.py`
  - paginated prediction history endpoint
- `api/routes/models.py`
  - active/recent model metadata endpoint

The API depends on the shared package rather than keeping separate featurization or model logic.

## Frontend Layer

The frontend is a single-page Next.js app in `web/`.

Homepage sections:

- Predict
  - SMILES input, example pills, prediction cards, descriptor grid
- History
  - active model strip
  - recent prediction distribution chart
  - paginated history table/cards backed by `/history` and `/models`
- Model Comparison
  - static metrics and charts sourced from `data/results.json`
- Methodology / footer

This split keeps interactive prediction and historical exploration separate while still fitting the app’s single-page structure.

## Runtime Data Flow

### Training

1. `scripts/train.py` parses CLI flags.
2. Alembic upgrades the DB.
3. `run_training_pipeline()` loads and featurizes ESOL.
4. Optuna performs CV tuning unless `--skip-tune` is used.
5. Final RF/NN models are retrained on the holdout split.
6. Models are logged to MLflow and registered in `model_versions`.
7. Local compatibility artifacts are written to `models/`.
8. Dashboard metrics are written to `data/results.json`.

### Prediction

1. User submits SMILES in the dashboard.
2. Frontend POSTs to `/predict`.
3. API featurizes the molecule with RDKit.
4. Loaded RF and NN models predict log(solubility).
5. API returns predictions plus descriptors.
6. API records the request in `predictions` best-effort.

### History / Registry

1. Frontend fetches `/history` and `/models`.
2. API queries SQLAlchemy repositories.
3. Frontend renders active model metadata, distribution summaries, and paginated rows.
