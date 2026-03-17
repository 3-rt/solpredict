"""Convert SMILES strings to molecular fingerprints and descriptors.

Morgan fingerprints are circular fingerprints that encode molecular substructure.
Each bit represents whether a particular chemical substructure (up to a given
radius from each atom) is present in the molecule. These are the standard
representation for ML on molecules.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def smiles_to_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Args:
        smiles: SMILES string representing a molecule (e.g., "CCO" for ethanol).
        radius: Radius of the circular fingerprint. radius=2 is roughly equivalent
                to ECFP4, a common choice in cheminformatics.
        n_bits: Length of the bit vector. 2048 gives a good balance of information
                content vs. dimensionality.

    Returns:
        numpy array of shape (n_bits,) with int8 values (0 or 1), or None if
        the SMILES string cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8)


def smiles_to_descriptors(smiles: str) -> dict | None:
    """Compute interpretable molecular descriptors from a SMILES string.

    These descriptors are human-readable properties that chemists use to
    understand and predict molecular behavior:
    - molecular_weight: Total mass of the molecule (daltons)
    - logp: Octanol-water partition coefficient — measures hydrophobicity.
            Higher LogP = less soluble in water (generally).
    - hbd: Number of hydrogen bond donors (e.g., OH, NH groups)
    - hba: Number of hydrogen bond acceptors (e.g., O, N atoms)
    - tpsa: Topological polar surface area — correlates with membrane
            permeability and solubility.

    Returns:
        Dictionary of descriptor values, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
    }
