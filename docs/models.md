# Model Details

## Dataset

SolPredict is trained on the ESOL (Delaney) dataset: 1,128 small organic molecules with experimentally measured aqueous solubility values expressed as `log(solubility)` in mol/L.

- target range is roughly `-11` to `+2`
- the default holdout split is 80/20
- reproducibility defaults to random seed `42`

## Feature Representation

Every molecule is converted into:

- a 2048-bit Morgan fingerprint (`radius=2`, ECFP4-style)
- an interpretable descriptor payload (`molecular_weight`, `logp`, `hbd`, `hba`, `tpsa`)

The fingerprint is the actual model input. Descriptors are used for API output and dashboard display, not for training.

## Random Forest

- library: scikit-learn
- feature input: raw Morgan fingerprints
- default no-tune path:
  - `n_estimators=200`
  - `max_features="sqrt"`
  - `min_samples_split=2`
  - `min_samples_leaf=1`

During a tuned run, Optuna searches over tree count, depth, split thresholds, leaf thresholds, and feature-subsampling strategy. The final trained version is stored in:

- local artifact: `models/random_forest.pkl`
- model registry row: `model_versions.name == "random_forest"`

## Neural Network

- library: PyTorch
- model class: `SolubilityMLP`
- configurable hidden layers and dropout
- default no-tune path:
  - `hidden_dims=(512, 128)`
  - `dropout=0.2`
  - `lr=1e-3`
  - `batch_size=64`
  - `weight_decay=1e-5`
  - `epochs=100`

During a tuned run, Optuna searches over hidden-layer layouts, dropout, learning rate, batch size, and weight decay. The API reconstructs the tuned architecture from the stored hyperparameters before loading weights.

Artifacts:

- local artifact: `models/neural_network.pt`
- model registry row: `model_versions.name == "neural_network"`

## Evaluation and Registry

Both models are evaluated with:

- `R²`
- `RMSE`
- `MAE`

Phase 2 added two places where those metrics live:

- `data/results.json`
  - powers the dashboard comparison and chart views
- `model_versions`
  - stores CV means, test metrics, trained timestamp, active flag, and MLflow run id

This avoids hard-coding a single metric snapshot in documentation. The latest values depend on the most recent training run.

## Tracking

Each final retraining run is logged to MLflow.

- RF uses `mlflow.sklearn.log_model`
- NN uses `mlflow.pytorch.log_model`
- the resulting run id is persisted in `model_versions.mlflow_run_id`

By default the tracking store is local:

```text
file://./mlruns
```

## Limitations

- ESOL is small, so both models remain data-limited
- fingerprints capture 2D structure only
- prediction quality is best for molecules similar to the training distribution
- the dashboard-compatible local artifacts are still filesystem based, even though the registry tracks version metadata separately
