from sqlalchemy import UniqueConstraint

from solpredict.db.models import Base, ModelVersion, Prediction


def test_metadata_registers_prediction_tables() -> None:
    table_names = set(Base.metadata.tables)
    assert table_names == {"model_versions", "predictions"}


def test_model_version_has_name_version_uniqueness() -> None:
    constraints = {
        tuple(constraint.columns.keys())
        for constraint in ModelVersion.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("name", "version") in constraints


def test_prediction_foreign_keys_target_model_versions() -> None:
    foreign_keys = {
        fk.parent.name: fk.column.table.name for fk in Prediction.__table__.foreign_keys
    }
    assert foreign_keys["rf_model_version_id"] == "model_versions"
    assert foreign_keys["nn_model_version_id"] == "model_versions"
