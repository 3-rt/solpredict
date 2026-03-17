import torch
import numpy as np
from solpredict.model import SolubilityMLP


def test_model_output_shape():
    """Model should take batch of fingerprints and return one prediction per sample."""
    model = SolubilityMLP(input_dim=2048)
    x = torch.randn(8, 2048)  # batch of 8
    out = model(x)
    assert out.shape == (8,)


def test_model_single_sample():
    """Model should handle a single sample."""
    model = SolubilityMLP(input_dim=2048)
    x = torch.randn(1, 2048)
    out = model(x)
    assert out.shape == (1,)


def test_model_deterministic_with_eval():
    """In eval mode (dropout disabled), same input should give same output."""
    model = SolubilityMLP(input_dim=2048)
    model.eval()
    x = torch.randn(4, 2048)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_model_custom_input_dim():
    """Model should work with different fingerprint sizes."""
    model = SolubilityMLP(input_dim=1024)
    x = torch.randn(4, 1024)
    out = model(x)
    assert out.shape == (4,)
