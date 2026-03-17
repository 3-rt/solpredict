"""Neural network model for solubility prediction.

A simple multi-layer perceptron (MLP) that takes Morgan fingerprints as input
and predicts log solubility. The architecture is intentionally simple:
three fully-connected layers with ReLU activation and dropout for regularization.

Architecture: 2048 → 512 → 128 → 1
- Input: 2048-bit Morgan fingerprint
- Hidden layers reduce dimensionality while learning non-linear relationships
- Dropout (20%) prevents overfitting on the small ESOL dataset
- Single output: predicted log(solubility) in mol/L
"""

import torch
import torch.nn as nn


class SolubilityMLP(nn.Module):
    """Multi-layer perceptron for predicting molecular solubility."""

    def __init__(self, input_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns shape (batch_size,)."""
        return self.net(x).squeeze(-1)
