from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from solpredict.db.engine import get_session_factory


def get_db() -> Iterator[Session]:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
