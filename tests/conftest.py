import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_raw(tmp_path):
    """A raw-layout tree (products/, metadata/) holding the mini fixture."""
    raw = tmp_path / "raw"
    (raw / "products").mkdir(parents=True)
    (raw / "metadata").mkdir()
    shutil.copy(FIXTURES / "mini_product.json", raw / "products" / "mini.json")
    shutil.copy(FIXTURES / "mini_metadata" / "employee.json", raw / "metadata" / "employee.json")
    shutil.copy(FIXTURES / "mini_metadata" / "customers_data.json", raw / "metadata" / "customers_data.json")
    (raw / "metadata" / "salesforce_team.json").write_text(json.dumps({"x": 1}))
    return raw


@pytest.fixture
def mini_bundle(mini_raw, tmp_path):
    from inherit.data.bundle import make_bundle

    out = tmp_path / "bundle"
    make_bundle(mini_raw, out)
    return out
