# SolPredict — Molecular Solubility Prediction

[![CI](https://github.com/3-rt/solpredict/actions/workflows/ci.yml/badge.svg)](https://github.com/3-rt/solpredict/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Predict aqueous solubility of organic molecules using a FastAPI backend, a Next.js dashboard, and a training pipeline built around RDKit, scikit-learn, PyTorch, Optuna, and MLflow.

## Overview

- `solpredict/` — shared package for featurization, models, config, DB access, training, and MLflow helpers
- `api/` — FastAPI app exposing prediction, history, and model-registry endpoints
- `web/` — single-page Next.js dashboard with prediction, history, and comparison sections
- `scripts/train.py` — CLI entry point for retraining and model registration
- `alembic/` — DB migrations for prediction history and model version metadata

## Quick Start

You need two terminals to run the API and dashboard locally. Training is a separate step you run first.

### 1. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test,api,notebook]"
```

### 2. Train models and register versions

```bash
source .venv/bin/activate
python3 scripts/train.py
```

Useful training variants:

```bash
# Fast smoke run for CI or local checks
python3 scripts/train.py --skip-tune --models rf --esol-csv tests/fixtures/esol_smoke.csv

# Retrain both models with a smaller Optuna budget
python3 scripts/train.py --n-trials 10
```

This command:

- runs Alembic migrations to `head`
- featurizes the ESOL dataset and caches features under `data/cache/`
- tunes or reuses default RF/NN hyperparameters
- logs training runs to MLflow
- writes local compatibility artifacts to `models/`
- updates `data/results.json`
- registers active model versions in the database

### 3. Start the API

```bash
source .venv/bin/activate
uvicorn api.main:app --port 7860
```

The API auto-runs Alembic migrations on startup unless `SOLPREDICT_SKIP_MIGRATIONS=1`.

### 4. Start the dashboard

```bash
cd web
npm install
npm run dev
```

The dashboard runs at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL=http://localhost:7860` if you want it to call a non-default API origin.

## Configuration

Application settings are loaded from environment variables or `.env` via `solpredict.config.Settings`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///solpredict.db` | API/training metadata DB |
| `MLFLOW_TRACKING_URI` | `file://./mlruns` | MLflow tracking store |
| `MODEL_DIR` | `./models` | Local serialized model artifacts |
| `DATA_DIR` | `./data` | Dataset/output root |
| `CACHE_DIR` | `./data/cache` | Cached featurization artifacts |
| `ESOL_CSV_PATH` | `./data/esol.csv` | Input dataset |
| `RESULTS_PATH` | `./data/results.json` | Dashboard metrics JSON |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `FP_RADIUS` | `2` | Morgan fingerprint radius |
| `FP_NBITS` | `2048` | Fingerprint width |
| `TRAIN_TEST_SIZE` | `0.2` | Holdout fraction |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `OPTUNA_TRIALS` | `50` | Default tuning budget |
| `MLP_EPOCHS` | `100` | Default MLP epochs |
| `LOG_LEVEL` | `INFO` | API logging level |
| `JSON_LOGS` | `false` | Structured log output |

Example `.env`:

```dotenv
DATABASE_URL=sqlite:///solpredict.db
MLFLOW_TRACKING_URI=file://./mlruns
LOG_LEVEL=INFO
JSON_LOGS=false
```

## Production Deployment

The production app is split across two hosts:

- Railway runs the FastAPI API and PostgreSQL database.
- Vercel runs the Next.js dashboard.

Deploy Railway first, then Vercel.

### Railway API

Required Railway API service variables:

```dotenv
DATABASE_URL=${{Postgres.DATABASE_URL}}
LOG_LEVEL=INFO
JSON_LOGS=false
```

Do not set `SOLPREDICT_SKIP_MIGRATIONS=1` in production. The API runs Alembic on startup
and creates the `model_versions`, `predictions`, and `alembic_version` tables automatically.

The API Docker image includes the bundled inference artifacts:

- `models/random_forest.pkl`
- `models/neural_network.pt`

Training is not run during deploy. Training only runs when `python3 scripts/train.py` is
executed explicitly.

### Vercel Dashboard

Required Vercel variable:

```dotenv
SOLPREDICT_API_URL=https://<railway-api-public-domain>
```

Use the Railway API base URL only. Do not include `/predict`, `/models`, or `/history`.

The dashboard calls same-origin routes (`/predict`, `/history`, `/models`) in production.
Those Next.js route handlers proxy to `SOLPREDICT_API_URL`, which avoids exposing backend
routing details in the browser bundle. `NEXT_PUBLIC_API_URL` is not required for the current
proxy setup.

### Production Smoke Tests

After Railway deploys:

```bash
curl https://<railway-api-public-domain>/health
curl https://<railway-api-public-domain>/models
curl "https://<railway-api-public-domain>/history?limit=10&offset=0"
```

After Vercel deploys:

```bash
curl https://<vercel-domain>/models
curl "https://<vercel-domain>/history?limit=10&offset=0"
curl -X POST https://<vercel-domain>/predict \
  -H "content-type: application/json" \
  --data '{"smiles":"CCO"}'
```

Expected `/health` model status:

```json
{
  "status": "ok",
  "models_loaded": {
    "random_forest": true,
    "neural_network": true
  }
}
```

## Database and Migrations

Alembic manages the application schema.

```bash
# Upgrade local DB to the latest schema
alembic upgrade head

# Inspect migration history
alembic history
```

Current schema includes:

- `model_versions` — registered RF/NN artifacts, metrics, hyperparameters, and active-version flags
- `predictions` — persisted `/predict` calls with descriptors, timestamps, and optional model-version foreign keys

If a fresh production database has no model registry rows, predictions can still work from
the bundled model files, but the dashboard will show active models as unavailable. See
[docs/deployment.md](docs/deployment.md) for the production seed SQL.

## Development

```bash
source .venv/bin/activate
pip install -e ".[dev,test,api,notebook]"
pre-commit install

ruff check api solpredict tests alembic scripts
ruff format --check api solpredict tests alembic scripts
mypy solpredict
pytest
pre-commit run --all-files
```

Frontend checks:

```bash
cd web
npm run lint
npm run build
```

## Key Docs

- [docs/architecture.md](docs/architecture.md)
- [docs/api-reference.md](docs/api-reference.md)
- [docs/deployment.md](docs/deployment.md)
- [docs/models.md](docs/models.md)
- [docs/ml-pipeline.md](docs/ml-pipeline.md)

## Project Structure

- `solpredict/featurize.py` — RDKit fingerprints and descriptors
- `solpredict/model.py` — configurable PyTorch MLP definition
- `solpredict/db/` — SQLAlchemy engine, models, repositories
- `solpredict/training/` — dataset loading, CV, tuning, orchestration
- `solpredict/tracking.py` — MLflow setup and model logging
- `api/routes/` — `/predict`, `/history`, `/models`
- `web/src/app/page.tsx` — single-page dashboard UI

## Outputs

- `models/random_forest.pkl`
- `models/neural_network.pt`
- `data/results.json`
- `mlruns/`
- `solpredict.db`
