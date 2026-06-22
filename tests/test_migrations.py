from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from api.main import run_startup_migrations
from solpredict.config import get_settings


def test_alembic_upgrade_head_creates_phase1_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "migration.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert {"model_versions", "predictions"} <= set(inspector.get_table_names())
    finally:
        engine.dispose()


def test_startup_migrations_use_database_url_env(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "startup.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.delenv("SOLPREDICT_SKIP_MIGRATIONS", raising=False)
    get_settings.cache_clear()

    try:
        run_startup_migrations()
    finally:
        get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert {"model_versions", "predictions"} <= set(inspector.get_table_names())
    finally:
        engine.dispose()
