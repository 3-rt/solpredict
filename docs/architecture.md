# Architecture

SolPredict has three layers: a shared Python library, a training pipeline, and a serving stack (API + frontend).

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Next.js     │────▶│  FastAPI      │────▶│  solpredict/  │
│  Dashboard   │     │  /predict     │     │  featurize.py │
│  (Vercel)    │     │  (HF Spaces)  │     │  model.py     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 ▲
                                                 │
                                          ┌──────┴───────┐
                                          │ scripts/      │
                                          │ train.py      │
                                          └──────────────┘
```

## Shared Library — `solpredict/`

Two modules used by both training and serving:

- **`featurize.py`** — Converts SMILES strings into ML-ready features:
  - Morgan fingerprints (2048-bit, radius=2, equivalent to ECFP4)
  - Molecular descriptors (MW, LogP, HBD, HBA, TPSA)
- **`model.py`** — Defines `SolubilityMLP`, a PyTorch MLP (2048 → 512 → 128 → 1) with ReLU and 20% dropout.

## Training Pipeline — `scripts/train.py`

Runs offline. Loads the ESOL CSV, featurizes all 1,128 molecules, splits 80/20, then trains two models:

| Model | Library | Test R² | Test RMSE |
|-------|---------|---------|-----------|
| Random Forest (100 trees) | scikit-learn | 0.71 | 1.17 |
| Neural Network (MLP) | PyTorch | 0.75 | 1.09 |

Exports:
- `models/random_forest.pkl` and `models/neural_network.pt` — serialized models for the API
- `data/results.json` — metrics, scatter plot data, residuals, and feature importance for the dashboard

## API — `api/`

FastAPI service that loads both models at startup via the lifespan context manager.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Takes `{"smiles": "..."}`, returns predictions from both models + molecular descriptors |
| `/health` | GET | Reports which models are loaded |
| `/examples` | GET | Returns predictions for 10 well-known molecules |

Deployed as a Docker container on HuggingFace Spaces.

## Frontend — `web/`

Next.js App Router with four pages:

| Page | Route | Content |
|------|-------|---------|
| Predict | `/` | SMILES input, example molecule pills, prediction results |
| Model Comparison | `/comparison` | Scatter plots, residual histograms, metrics table |
| Methodology | `/methodology` | Explanation of dataset, fingerprints, models, metrics |
| About | `/about` | Project motivation, tech stack, links |

Fetches predictions from the API at runtime. Loads `results.json` statically for the comparison page.

## Data Flow

1. User enters a SMILES string in the dashboard
2. Frontend POSTs to `/predict`
3. API converts SMILES → Morgan fingerprint via RDKit
4. Both models predict log(solubility) from the fingerprint
5. API also computes molecular descriptors for display
6. Frontend renders predictions and descriptor cards
