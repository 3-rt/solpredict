"""Custom exception hierarchy for SolPredict."""

from __future__ import annotations


class SolPredictError(Exception):
    """Base class for all SolPredict errors."""


class InvalidSmilesError(SolPredictError):
    """Raised when a SMILES string cannot be parsed by RDKit."""

    def __init__(self, smiles: str) -> None:
        super().__init__(f"Could not parse SMILES: {smiles!r}")
        self.smiles = smiles


class ModelNotLoadedError(SolPredictError):
    """Raised when an API endpoint is invoked before a model is loaded."""

    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model not loaded: {model_name!r}")
        self.model_name = model_name


class ModelVersionNotFoundError(SolPredictError):
    """Raised when a requested model_versions row does not exist."""

    def __init__(self, name: str, version: str | None = None) -> None:
        detail = f"{name!r}" if version is None else f"{name!r} version {version!r}"
        super().__init__(f"Model version not found: {detail}")
        self.name = name
        self.version = version
