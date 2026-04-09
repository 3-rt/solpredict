import numpy as np
from solpredict.featurize import smiles_to_fingerprint, smiles_to_descriptors


def test_fingerprint_valid_smiles():
    """Ethanol (CCO) should produce a 2048-bit fingerprint."""
    fp = smiles_to_fingerprint("CCO")
    assert fp is not None
    assert fp.shape == (2048,)
    assert fp.dtype == np.int8
    assert fp.sum() > 0  # at least some bits set


def test_fingerprint_invalid_smiles():
    """Invalid SMILES should return None."""
    assert smiles_to_fingerprint("NOT_A_MOLECULE") is None


def test_fingerprint_custom_params():
    """Should respect radius and n_bits parameters."""
    fp = smiles_to_fingerprint("CCO", radius=3, n_bits=1024)
    assert fp.shape == (1024,)


def test_fingerprint_does_not_emit_rdkit_deprecation_warning(capfd):
    """Fingerprint generation should not use RDKit's deprecated Morgan API."""
    fp = smiles_to_fingerprint("CCO")
    captured = capfd.readouterr()

    assert fp is not None
    assert "please use MorganGenerator" not in captured.err


def test_descriptors_valid_smiles():
    """Ethanol descriptors should be chemically reasonable."""
    desc = smiles_to_descriptors("CCO")
    assert desc is not None
    assert set(desc.keys()) == {"molecular_weight", "logp", "hbd", "hba", "tpsa"}
    # Ethanol: MW ≈ 46.07, 1 OH group
    assert 45 < desc["molecular_weight"] < 47
    assert desc["hbd"] == 1  # one OH donor
    assert desc["hba"] == 1  # one oxygen acceptor


def test_descriptors_invalid_smiles():
    """Invalid SMILES should return None."""
    assert smiles_to_descriptors("INVALID") is None
