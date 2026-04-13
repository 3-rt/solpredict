import os
from pathlib import Path

import pytest

from solpredict.config import Settings, get_settings


def test_defaults_are_sensible(monkeypatch):
    # Clear any env vars that would override defaults
    for key in ["DATABASE_URL", "MLFLOW_TRACKING_URI", "RANDOM_SEED", "LOG_LEVEL"]:
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.random_seed == 42
    assert s.fp_radius == 2
    assert s.fp_nbits == 2048
    assert s.log_level == "INFO"
    assert s.database_url.startswith("sqlite:///")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RANDOM_SEED", "1337")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    s = Settings(_env_file=None)
    assert s.random_seed == 1337
    assert s.log_level == "DEBUG"
    assert s.database_url == "postgresql://u:p@h/db"


def test_log_level_is_validated(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_get_settings_is_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_paths_are_absolute():
    s = Settings(_env_file=None)
    assert Path(s.model_dir).is_absolute()
    assert Path(s.data_dir).is_absolute()
