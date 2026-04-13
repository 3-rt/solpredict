"""Application settings loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Database
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'solpredict.db'}"

    # Experiment tracking
    mlflow_tracking_uri: str = f"file://{PROJECT_ROOT / 'mlruns'}"

    # Filesystem paths (absolute)
    model_dir: str = str(PROJECT_ROOT / "models")
    data_dir: str = str(PROJECT_ROOT / "data")
    cache_dir: str = str(PROJECT_ROOT / "data" / "cache")

    # ML hyperparameters / reproducibility
    random_seed: int = 42
    fp_radius: int = 2
    fp_nbits: int = 2048

    # Logging
    log_level: LogLevel = "INFO"
    json_logs: bool = False

    @field_validator("model_dir", "data_dir", "cache_dir")
    @classmethod
    def _absolutize(cls, v: str) -> str:
        p = Path(v)
        return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
