import json
import logging

import pytest

import solpredict.logging_setup as ls_module
from solpredict.logging_setup import configure_logging


@pytest.fixture(autouse=True)
def _reset_logging_flag():
    ls_module._CONFIGURED = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    ls_module._CONFIGURED = False
    for h in list(root.handlers):
        root.removeHandler(h)


def test_configure_logging_sets_root_level(caplog):
    configure_logging(level="DEBUG", json_logs=False)
    logger = logging.getLogger("solpredict.test")
    assert logger.getEffectiveLevel() == logging.DEBUG


def test_configure_logging_human_format(capsys):
    configure_logging(level="INFO", json_logs=False)
    logging.getLogger("solpredict.test").info("hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.err
    # Human format should not be valid JSON
    for line in captured.err.strip().splitlines():
        try:
            json.loads(line)
        except json.JSONDecodeError:
            return  # expected
    raise AssertionError("expected at least one non-JSON line")


def test_configure_logging_json_format(capsys):
    configure_logging(level="INFO", json_logs=True)
    logging.getLogger("solpredict.test").info("structured", extra={"smiles": "CCO"})
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["message"] == "structured"
    assert data["level"] == "INFO"
    assert data["smiles"] == "CCO"


def test_configure_logging_is_idempotent():
    configure_logging(level="INFO", json_logs=False)
    handlers_before = list(logging.getLogger().handlers)
    configure_logging(level="INFO", json_logs=False)
    handlers_after = list(logging.getLogger().handlers)
    assert len(handlers_after) == len(handlers_before)
