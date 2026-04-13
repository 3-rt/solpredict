"""Create prediction history and model version tables."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260412_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("cv_r2_mean", sa.Float(), nullable=True),
        sa.Column("cv_rmse_mean", sa.Float(), nullable=True),
        sa.Column("test_r2", sa.Float(), nullable=True),
        sa.Column("test_rmse", sa.Float(), nullable=True),
        sa.Column("hyperparameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
    )
    op.create_index("ix_model_versions_name", "model_versions", ["name"], unique=False)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("smiles", sa.String(length=512), nullable=False),
        sa.Column("rf_prediction", sa.Float(), nullable=True),
        sa.Column("nn_prediction", sa.Float(), nullable=True),
        sa.Column(
            "rf_model_version_id",
            sa.Integer(),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column(
            "nn_model_version_id",
            sa.Integer(),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column("descriptors", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("molecule_name", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("client_ip", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_predictions_smiles", "predictions", ["smiles"], unique=False)
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_smiles", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_model_versions_name", table_name="model_versions")
    op.drop_table("model_versions")
