from pathlib import Path

import pandas as pd

from solpredict.config import Settings
from solpredict.training.data import build_feature_cache_key, featurize_dataset, load_esol


def test_training_settings_defaults_are_absolute() -> None:
    settings = Settings(_env_file=None)
    assert settings.train_test_size == 0.2
    assert settings.cv_folds == 5
    assert settings.optuna_trials == 50
    assert Path(settings.esol_csv_path).is_absolute()
    assert settings.results_path.endswith("data/results.json")


def test_load_esol_normalizes_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "esol.csv"
    csv_path.write_text(
        "Compound ID,smiles,measured log solubility in mols per litre\nethanol,CCO,-0.3\n",
        encoding="utf-8",
    )
    frame = load_esol(csv_path)
    assert list(frame.columns) == ["smiles", "log_solubility", "name"]
    assert frame.loc[0, "smiles"] == "CCO"


def test_featurize_dataset_uses_cache(tmp_path: Path) -> None:
    frame = pd.DataFrame({"smiles": ["CCO"], "log_solubility": [-0.3]})
    cache_dir = tmp_path / "cache"
    first = featurize_dataset(frame, cache_dir=cache_dir, fp_radius=2, fp_nbits=2048)
    second = featurize_dataset(frame, cache_dir=cache_dir, fp_radius=2, fp_nbits=2048)
    assert first.fingerprints.shape == (1, 2048)
    assert second.cache_hit is True
    assert build_feature_cache_key(frame, fp_radius=2, fp_nbits=2048)
