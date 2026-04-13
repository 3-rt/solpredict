# ML Pipeline

This document describes the Phase 2 training system that replaced the original monolithic notebook/script flow.

## Entry Point

Use the CLI in `scripts/train.py`:

```bash
python3 scripts/train.py
```

Useful flags:

```bash
python3 scripts/train.py --skip-tune
python3 scripts/train.py --models rf
python3 scripts/train.py --n-trials 10
python3 scripts/train.py --esol-csv tests/fixtures/esol_smoke.csv
```

Before training begins, the script upgrades the DB with Alembic so model-version registration always has the expected schema.

## Pipeline Stages

The main orchestration lives in `solpredict/training/pipeline.py`.

### 1. Load dataset

`training.data.load_esol()`:

- reads the ESOL CSV
- normalizes column names
- keeps `smiles`, `log_solubility`, and optional `name`

### 2. Featurize and cache

`training.data.featurize_dataset()`:

- generates fingerprints and descriptors with RDKit
- drops invalid molecules from the supervised training arrays
- caches feature arrays in `data/cache/` using a content-derived cache key

This keeps repeated local runs and smoke runs cheaper.

### 3. Holdout split

`training.data.split_holdout()` performs the train/test split using the configured seed and `TRAIN_TEST_SIZE`.

### 4. Tune hyperparameters

`training.tune.tune_models()` handles model selection for RF and NN.

Two modes:

- default mode
  - runs Optuna
  - evaluates candidates with 5-fold CV
  - records metrics to MLflow
- `--skip-tune`
  - uses stable default parameter sets
  - still computes CV metrics for the selected defaults

RF search space includes:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

NN search space includes:

- hidden-layer layout
- dropout
- learning rate
- batch size
- weight decay

## 5. Retrain final models

Once tuning picks the best parameter sets, the pipeline retrains final RF and NN models on the training split and evaluates them on the holdout split.

Artifacts written locally:

- `models/random_forest.pkl`
- `models/neural_network.pt`

Dashboard metrics written locally:

- `data/results.json`

## 6. Log to MLflow

`solpredict/tracking.py` configures the tracking URI and logs final trained models:

- RF via `mlflow.sklearn.log_model`
- NN via `mlflow.pytorch.log_model`

The default local tracking URI is:

```text
file://./mlruns
```

## 7. Register model versions

Each successful final model is inserted or updated through `upsert_model_version(...)`.

Stored metadata includes:

- `name`
- `version`
- `artifact_path`
- `mlflow_run_id`
- `trained_at`
- CV summary metrics
- holdout test metrics
- hyperparameters
- `is_active`

The API uses the active registry rows to know which artifacts to load at startup.

## Outputs

After a successful run you should have:

- local serialized artifacts in `models/`
- MLflow runs in `mlruns/`
- updated dashboard metrics in `data/results.json`
- active registry rows in `model_versions`

## Smoke Path

For CI and quick local checks, use:

```bash
python3 scripts/train.py --skip-tune --models rf --esol-csv tests/fixtures/esol_smoke.csv
```

That keeps the run cheap while still exercising:

- migration setup
- data loading
- cached featurization
- retraining/export
- model registration

## Notes

- The pipeline still writes local compatibility artifacts because the API loads from filesystem paths today.
- The registry and MLflow tracking make those artifacts versioned and inspectable, even though deployment still uses the local copies.
