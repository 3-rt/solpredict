# API Reference

Base URL (local): `http://localhost:7860`

## POST /predict

Predict aqueous solubility for a molecule given its SMILES string.

**Request:**
```json
{
  "smiles": "CCO"
}
```

**Response (valid molecule):**
```json
{
  "smiles": "CCO",
  "valid": true,
  "predictions": {
    "random_forest": -0.7312,
    "neural_network": -0.5841
  },
  "descriptors": {
    "molecular_weight": 46.07,
    "logp": -0.0014,
    "hbd": 1,
    "hba": 1,
    "tpsa": 20.23
  },
  "molecule_name": "Ethanol"
}
```

**Response (invalid SMILES):**
```json
{
  "smiles": "not_a_molecule",
  "valid": false,
  "error": "Could not parse SMILES string"
}
```

`molecule_name` is non-null only for 10 well-known molecules (Ethanol, Aspirin, Caffeine, Glucose, Ibuprofen, Benzene, Acetic acid, Water, Dodecane, Anthracene).

Prediction values are log(solubility) in mol/L. More negative = less soluble.

## GET /health

**Response:**
```json
{
  "status": "ok",
  "models_loaded": {
    "random_forest": true,
    "neural_network": true
  }
}
```

## GET /examples

Returns predictions for 10 well-known molecules. Useful for populating the dashboard without user input.

**Response:** Array of objects with the same shape as `/predict` responses, plus a `name` field.

## Running Locally

```bash
# From project root
uvicorn api.main:app --port 7860
```

Models must exist at `models/random_forest.pkl` and `models/neural_network.pt`. Run `python scripts/train.py` first to generate them.
