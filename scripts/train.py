#!/usr/bin/env python3
"""Training pipeline for molecular solubility prediction.

Trains a Random Forest and a Neural Network on the ESOL dataset,
evaluates both models, and exports artifacts for the prediction API
and web dashboard.

Usage:
    python scripts/train.py

Outputs:
    models/random_forest.pkl   — Trained Random Forest model
    models/neural_network.pt   — Trained Neural Network state dict
    data/results.json          — Metrics, plot data, and model comparison
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add project root to path so we can import solpredict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solpredict.featurize import smiles_to_descriptors, smiles_to_fingerprint
from solpredict.model import SolubilityMLP

# ── Configuration ──────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
FP_RADIUS = 2
FP_NBITS = 2048
NN_EPOCHS = 100
NN_LR = 0.001
NN_BATCH_SIZE = 64

DATA_PATH = "data/esol.csv"
RF_MODEL_PATH = "models/random_forest.pkl"
NN_MODEL_PATH = "models/neural_network.pt"
RESULTS_PATH = "data/results.json"


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the ESOL dataset."""
    df = pd.read_csv(path)
    # Standardize column names (handle varying capitalization across dataset versions)
    df.columns = df.columns.str.strip()
    col_map = {}
    for col in df.columns:
        if col.lower() == "smiles":
            col_map[col] = "smiles"
        elif "measured log solubility" in col.lower():
            col_map[col] = "log_solubility"
        elif col.lower() in ("compound id", "compound_id", "name"):
            col_map[col] = "name"
    df = df.rename(columns=col_map)
    # Keep only what we need
    cols_to_keep = ["smiles", "log_solubility"]
    if "name" in df.columns:
        cols_to_keep.append("name")
    df = df[cols_to_keep]
    print(f"Loaded {len(df)} molecules from {path}")
    return df


def featurize_dataset(df: pd.DataFrame) -> tuple:
    """Generate fingerprints and descriptors for all molecules."""
    fingerprints = []
    descriptors = []
    valid_mask = []

    for smiles in df["smiles"]:
        fp = smiles_to_fingerprint(smiles, radius=FP_RADIUS, n_bits=FP_NBITS)
        desc = smiles_to_descriptors(smiles)
        if fp is not None and desc is not None:
            fingerprints.append(fp)
            descriptors.append(desc)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"Warning: {n_invalid} molecules could not be parsed and were skipped")

    return np.array(fingerprints), descriptors, valid_mask


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest regressor."""
    print("\n── Training Random Forest ──")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)

    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)
    print(f"  Train — R²: {train_metrics['r2']}, RMSE: {train_metrics['rmse']}")
    print(f"  Test  — R²: {test_metrics['r2']}, RMSE: {test_metrics['rmse']}")

    return rf, test_pred, train_metrics, test_metrics


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train and evaluate a simple MLP."""
    print("\n── Training Neural Network ──")
    torch.manual_seed(RANDOM_SEED)

    X_train_t = torch.FloatTensor(X_train.astype(np.float32))
    y_train_t = torch.FloatTensor(y_train.astype(np.float32))
    X_test_t = torch.FloatTensor(X_test.astype(np.float32))

    model = SolubilityMLP(input_dim=FP_NBITS)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)
    loss_fn = nn.MSELoss()

    model.train()
    training_losses = []
    for epoch in range(NN_EPOCHS):
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), NN_BATCH_SIZE):
            batch_idx = indices[i : i + NN_BATCH_SIZE]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        training_losses.append(round(avg_loss, 4))
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{NN_EPOCHS} — Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).numpy()
        test_pred = model(X_test_t).numpy()

    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)
    print(f"  Train — R²: {train_metrics['r2']}, RMSE: {train_metrics['rmse']}")
    print(f"  Test  — R²: {test_metrics['r2']}, RMSE: {test_metrics['rmse']}")

    return model, test_pred, train_metrics, test_metrics, training_losses


def export_results(
    y_test, rf_pred, nn_pred,
    rf_train_metrics, rf_test_metrics,
    nn_train_metrics, nn_test_metrics,
    nn_training_losses, rf_model, feature_names=None
):
    """Export all results as JSON for the web dashboard."""
    results = {
        "dataset": {
            "name": "ESOL (Delaney)",
            "n_molecules": 1128,
            "target": "log(solubility) in mol/L",
            "split": f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test",
            "random_seed": RANDOM_SEED,
        },
        "models": {
            "random_forest": {
                "name": "Random Forest",
                "library": "scikit-learn",
                "params": {"n_estimators": 100},
                "train_metrics": rf_train_metrics,
                "test_metrics": rf_test_metrics,
            },
            "neural_network": {
                "name": "Neural Network",
                "library": "PyTorch",
                "architecture": "2048 → 512 → 128 → 1",
                "params": {
                    "epochs": NN_EPOCHS,
                    "learning_rate": NN_LR,
                    "batch_size": NN_BATCH_SIZE,
                    "dropout": 0.2,
                },
                "train_metrics": nn_train_metrics,
                "test_metrics": nn_test_metrics,
                "training_losses": nn_training_losses,
            },
        },
        "plots": {
            "scatter": {
                "y_true": y_test.tolist(),
                "rf_pred": rf_pred.tolist(),
                "nn_pred": nn_pred.tolist(),
            },
            "residuals": {
                "rf_residuals": (y_test - rf_pred).tolist(),
                "nn_residuals": (y_test - nn_pred).tolist(),
            },
        },
    }

    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        top_indices = np.argsort(importances)[-20:][::-1]
        results["feature_importance"] = {
            "bit_positions": top_indices.tolist(),
            "importances": importances[top_indices].tolist(),
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {RESULTS_PATH}")


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("SolPredict — Molecular Solubility Prediction")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print("\nGenerating molecular fingerprints and descriptors...")
    fingerprints, descriptors, valid_mask = featurize_dataset(df)
    y = df.loc[valid_mask, "log_solubility"].values

    X_train, X_test, y_train, y_test = train_test_split(
        fingerprints, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"\nTrain: {len(X_train)} molecules, Test: {len(X_test)} molecules")

    rf_model, rf_pred, rf_train, rf_test = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    nn_model, nn_pred, nn_train, nn_test, nn_losses = train_neural_network(
        X_train, y_train, X_test, y_test
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, RF_MODEL_PATH)
    torch.save(nn_model.state_dict(), NN_MODEL_PATH)
    print(f"\nModels saved to {RF_MODEL_PATH} and {NN_MODEL_PATH}")

    export_results(
        y_test, rf_pred, nn_pred,
        rf_train, rf_test, nn_train, nn_test,
        nn_losses, rf_model,
    )

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
