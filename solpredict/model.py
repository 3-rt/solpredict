"""Neural network model for solubility prediction.

A simple multi-layer perceptron (MLP) that takes Morgan fingerprints as input
and predicts log solubility. The architecture is parameterized by ``input_dim``
and ``hidden_dims``: for each hidden dim the model stacks Linear -> ReLU ->
Dropout, then a final Linear projection to a scalar output.

Architecture: input_dim -> *hidden_dims -> 1
- Input: ``input_dim`` (default 2048) Morgan fingerprint bits
- Hidden layers: configurable via ``hidden_dims`` tuple (default ``(512, 128)``)
- Dropout (default 20%) prevents overfitting on the small ESOL dataset
- Single output: predicted log(solubility) in mol/L
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SolubilityMLP(nn.Module):
    """Multi-layer perceptron for predicting molecular solubility."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: tuple[int, ...] = (512, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns shape (batch_size,)."""
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out
