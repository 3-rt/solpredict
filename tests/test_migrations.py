from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command


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
