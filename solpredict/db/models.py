from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_versions_name_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(64))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(512))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    cv_r2_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    cv_rmse_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    smiles: Mapped[str] = mapped_column(String(512), index=True)
    rf_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    nn_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    rf_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=True,
    )
    nn_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=True,
    )
    descriptors: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    molecule_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    rf_model_version: Mapped[ModelVersion | None] = relationship(foreign_keys=[rf_model_version_id])
    nn_model_version: Mapped[ModelVersion | None] = relationship(foreign_keys=[nn_model_version_id])
