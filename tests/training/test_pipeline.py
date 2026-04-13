from pathlib import Path

import mlflow
from sqlalchemy.orm import Session

from scripts.train import build_parser
from solpredict.db.models import ModelVersion
from solpredict.tracking import configure_mlflow_tracking
from solpredict.training.pipeline import run_training_pipeline


def test_configure_mlflow_tracking_creates_local_store(tmp_path: Path) -> None:
    uri = f"file://{tmp_path / 'mlruns'}"
    configure_mlflow_tracking(uri)
    assert mlflow.get_tracking_uri() == uri


def test_run_training_pipeline_writes_artifacts_results_and_db_rows(
    db_engine,
    training_tmp_dirs,
) -> None:
    csv_path = Path("tests/fixtures/esol_smoke.csv")
    with Session(db_engine, expire_on_commit=False) as session:
        outcome = run_training_pipeline(
            db_session=session,
            esol_csv_path=csv_path,
            models=("rf", "nn"),
            skip_tune=True,
            n_trials=2,
        )
        assert outcome.results_path.exists()
        assert outcome.rf_model_path.exists()
        assert outcome.nn_model_path.exists()
        assert session.query(ModelVersion).count() == 2


def test_build_parser_supports_skip_tune_and_model_selection() -> None:
    parser = build_parser()
    args = parser.parse_args(["--skip-tune", "--models", "rf", "nn", "--n-trials", "7"])
    assert args.skip_tune is True
    assert args.models == ["rf", "nn"]
    assert args.n_trials == 7


def test_run_training_pipeline_supports_rf_only_skip_tune(
    db_engine,
    training_tmp_dirs,
) -> None:
    csv_path = Path("tests/fixtures/esol_smoke.csv")
    with Session(db_engine, expire_on_commit=False) as session:
        outcome = run_training_pipeline(
            db_session=session,
            esol_csv_path=csv_path,
            models=("rf",),
            skip_tune=True,
            n_trials=2,
        )
        assert outcome.rf_model_path.exists()
        assert outcome.results_path.exists()
