import pytest

from solpredict.exceptions import (
    InvalidSmilesError,
    ModelNotLoadedError,
    ModelVersionNotFoundError,
    SolPredictError,
)


def test_all_inherit_from_base():
    assert issubclass(InvalidSmilesError, SolPredictError)
    assert issubclass(ModelNotLoadedError, SolPredictError)
    assert issubclass(ModelVersionNotFoundError, SolPredictError)


def test_invalid_smiles_carries_input():
    err = InvalidSmilesError("NOT_VALID")
    assert err.smiles == "NOT_VALID"
    assert "NOT_VALID" in str(err)


def test_model_not_loaded_carries_name():
    err = ModelNotLoadedError("random_forest")
    assert err.model_name == "random_forest"
    assert "random_forest" in str(err)


def test_model_version_not_found_carries_details():
    err = ModelVersionNotFoundError(name="rf", version="2026.04.12-rf")
    assert err.name == "rf"
    assert err.version == "2026.04.12-rf"
    assert "2026.04.12-rf" in str(err)


def test_raise_and_catch_via_base():
    with pytest.raises(SolPredictError):
        raise InvalidSmilesError("X")
