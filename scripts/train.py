#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alembic.config import Config

from alembic import command
from solpredict.db.engine import get_session_factory
from solpredict.training.pipeline import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SolPredict models.")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--models", nargs="+", choices=["rf", "nn"], default=["rf", "nn"])
    parser.add_argument("--esol-csv", default=None)
    return parser


def run_migrations() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")


def main() -> None:
    args = build_parser().parse_args()
    run_migrations()
    session_factory = get_session_factory()
    with session_factory() as session:
        run_training_pipeline(
            db_session=session,
            esol_csv_path=args.esol_csv,
            models=tuple(args.models),
            skip_tune=args.skip_tune,
            n_trials=args.n_trials,
        )


if __name__ == "__main__":
    main()
