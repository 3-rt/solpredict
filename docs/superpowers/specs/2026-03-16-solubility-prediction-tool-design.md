# Molecular Solubility Prediction Tool — Design Spec

## Overview

A portfolio project demonstrating CS + Chemistry competence through a molecular solubility prediction tool. Uses the ESOL dataset, RDKit for molecular featurization, and compares Random Forest vs Neural Network performance. Deployed as a polished web dashboard on Vercel with a live prediction API on HuggingFace Spaces.

**Target audience:** Academic reviewers (grad school admissions) and industry recruiters. Must demonstrate both research rigor and solid software engineering.

**Constraint:** Buildable in one night. Favor simplicity and pre-computed results over complex infrastructure.

## Architecture

Monorepo with three components:

```
chem-project/
├── notebooks/    — Training & analysis (Jupyter)
├── api/          — Prediction service (FastAPI on HuggingFace Spaces)
├── web/          — Dashboard (Next.js on Vercel)
├── models/       — Saved model artifacts (RF pickle, NN state_dict)
└── data/         — ESOL dataset + processed features
```

### Data flow

1. **Training (offline):** Notebook loads ESOL → RDKit generates fingerprints → trains both models → exports artifacts + metrics JSON
2. **Prediction (live):** Dashboard sends SMILES to API → API generates fingerprint with RDKit → runs both models → returns predictions
3. **Static content:** Dashboard loads pre-exported metrics/plots JSON for the Model Comparison and Methodology pages

## Component 1: Training Pipeline

**Location:** `notebooks/training.ipynb`

### Dataset

ESOL (Estimated SOLubility) — ~1,128 molecules with experimentally measured log solubility (log mol/L) in water. Standard cheminformatics benchmark. Included as CSV in `data/` or pulled from `deepchem`.

### Feature generation

- RDKit parses SMILES → molecule objects
- Morgan fingerprints: radius=2, 2048 bits (circular fingerprints encoding molecular substructure)
- Interpretable descriptors: molecular weight, LogP, number of H-bond donors/acceptors, topological polar surface area (TPSA)

### Models

**Random Forest (sklearn):**
- `n_estimators=100`, default hyperparameters
- Fast to train, interpretable via feature importance
- Serves as a strong baseline

**Neural Network (PyTorch):**
- 3-layer MLP: 2048 → 512 → 128 → 1
- ReLU activations, dropout (0.2), Adam optimizer
- Simple enough to explain, complex enough to be interesting

**Training:**
- 80/20 train/test split, fixed random seed for reproducibility
- No hyperparameter sweeps — straightforward training

### Evaluation

- Metrics: R², RMSE, MAE for both models
- Predicted vs actual scatter plots
- Residual distribution plots
- Feature importance (RF)
- All metrics and plot data exported as JSON to `data/results.json`

### Notebook style

Structured like a research notebook with markdown sections: Introduction, Data Exploration, Feature Engineering, Model Training, Results, Conclusion. Heavy comments explaining the chemistry (why Morgan fingerprints work, what solubility means physically, what the descriptors capture).

## Component 2: Prediction API

**Location:** `api/`
**Deployment:** HuggingFace Spaces (Docker, free tier)
**Framework:** FastAPI

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Takes `{"smiles": "CCO"}`, returns predictions from both models + molecular descriptors |
| `GET` | `/health` | Healthcheck for dashboard connectivity |
| `GET` | `/examples` | Returns ~10 curated molecules with pre-computed predictions |

### `/predict` response shape

```json
{
  "smiles": "CCO",
  "valid": true,
  "predictions": {
    "random_forest": -0.74,
    "neural_network": -0.81
  },
  "descriptors": {
    "molecular_weight": 46.07,
    "logp": -0.18,
    "hbd": 1,
    "hba": 1,
    "tpsa": 20.23
  },
  "molecule_name": "Ethanol"
}
```

### Cold start mitigation

HuggingFace free tier sleeps after ~15 min inactivity. Mitigations:
- Dashboard loads pre-computed example results by default (from static JSON bundled in the frontend)
- API calls are lazy — triggered only when user submits a SMILES string
- Loading state shows "Waking up the prediction server..." with a progress indicator
- If API is unreachable after timeout, display pre-computed fallback gracefully

### Dependencies

RDKit, FastAPI, scikit-learn, torch (CPU), uvicorn. Dockerfile for HuggingFace Spaces.

## Component 3: Web Dashboard

**Location:** `web/`
**Deployment:** Vercel
**Tech:** Next.js (App Router), Tailwind CSS, Recharts

### Design

Dark theme, clean and minimal. Scientific but modern — professional research tool aesthetic, not homework.

### Pages

**1. Predict (home page)**
- SMILES text input with "Predict" button
- Example molecule pills below input: Aspirin, Caffeine, Ethanol, Glucose, Ibuprofen
- Side-by-side prediction cards: RF (blue accent) vs NN (purple accent), showing predicted log solubility and model metrics
- Molecular properties panel: weight, LogP, H-bond donors/acceptors in a grid

**2. Model Comparison**
- Predicted vs actual scatter plots for both models (Recharts)
- Residual distribution charts
- Metrics comparison table (R², RMSE, MAE)
- All data loaded from static JSON exported during training

**3. Methodology**
- How Morgan fingerprints work (with diagrams)
- Model architecture descriptions
- Dataset information and preprocessing steps
- The "research paper" page — demonstrates chemistry domain knowledge

**4. About**
- Project motivation and background
- Links to GitHub repo and training notebook
- Tech stack overview

### API integration

- `POST /predict` called on form submission
- Loading state during cold start with user-friendly messaging
- Pre-computed example results bundled as static JSON fallback
- Error handling: invalid SMILES feedback, API timeout graceful degradation

## Deployment

### Vercel (frontend)
- Connect GitHub repo, set root directory to `web/`
- Environment variable: `NEXT_PUBLIC_API_URL` pointing to HuggingFace Spaces URL

### HuggingFace Spaces (API)
- Docker Space with `Dockerfile` in `api/`
- Model artifacts committed to the Space or loaded from the repo
- Free tier, CPU-only (sufficient for this scale)

## What's out of scope

- Custom model architectures or hyperparameter optimization
- User accounts or saved predictions
- Molecular structure 2D/3D visualization (nice-to-have for v2)
- Database or persistent storage
- CI/CD beyond Vercel/HF auto-deploy from GitHub
