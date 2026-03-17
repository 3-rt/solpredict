# Model Details

## Dataset

**ESOL (Delaney)** — 1,128 organic molecules with experimentally measured aqueous solubility values. The target variable is log(solubility) in mol/L, ranging from roughly -11 to +2.

Source: Delaney, J.S. "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure." *J. Chem. Inf. Comput. Sci.* 2004, 44, 1000-1005.

Split: 80% train (902 molecules), 20% test (226 molecules), random seed 42.

## Feature Representation

Each molecule is represented as a **2048-bit Morgan fingerprint** (radius=2), which is equivalent to ECFP4 — a standard circular fingerprint in cheminformatics. Each bit indicates the presence or absence of a particular molecular substructure within a 2-bond radius of each atom.

Morgan fingerprints are the input to both models. Molecular descriptors (MW, LogP, etc.) are computed separately for interpretability but are not used as model features.

## Random Forest

- **Library:** scikit-learn
- **Configuration:** 100 trees, default hyperparameters
- **Input:** Raw 2048-bit fingerprint vectors

| Split | R² | RMSE | MAE |
|-------|-----|------|-----|
| Train | 0.94 | 0.51 | 0.35 |
| Test | 0.71 | 1.17 | 0.88 |

The gap between train and test R² (0.94 vs 0.71) suggests some overfitting, which is expected for Random Forest on a small dataset without hyperparameter tuning.

## Neural Network

- **Library:** PyTorch
- **Architecture:** Linear(2048→512) → ReLU → Dropout(0.2) → Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→1)
- **Optimizer:** Adam (lr=0.001)
- **Training:** 100 epochs, batch size 64, MSE loss

| Split | R² | RMSE | MAE |
|-------|-----|------|-----|
| Train | 0.98 | 0.32 | 0.16 |
| Test | 0.75 | 1.09 | 0.79 |

The MLP slightly outperforms Random Forest on the test set (R² 0.75 vs 0.71). Both models show the train-test gap typical of small datasets.

## Evaluation Metrics

- **R² (coefficient of determination):** Proportion of variance explained. 1.0 is perfect; 0.0 means the model is no better than predicting the mean.
- **RMSE (root mean squared error):** Average prediction error in log(mol/L). Penalizes large errors more than MAE.
- **MAE (mean absolute error):** Average absolute prediction error. More robust to outliers than RMSE.

## Limitations

- Small dataset (1,128 molecules) limits generalization
- No hyperparameter tuning — both models use reasonable defaults
- Morgan fingerprints lose 3D structural information
- Predictions are most reliable for molecules similar to the ESOL training set (small organic druglike molecules)
