import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("run_dev", Path(__file__).resolve().parents[1] / "scripts" / "run_dev.py")
run_dev = importlib.util.module_from_spec(_spec)
sys.modules["run_dev"] = run_dev
_spec.loader.exec_module(run_dev)


def test_held_product_is_refused_before_any_question_is_loaded(monkeypatch):
    monkeypatch.setattr(run_dev, "load_questions", lambda *a, **k: pytest.fail("questions were loaded"))
    with pytest.raises(SystemExit, match="refusing non-dev products"):
        run_dev.build_questions(["ActionGenie", "SomeHeldProduct"])
