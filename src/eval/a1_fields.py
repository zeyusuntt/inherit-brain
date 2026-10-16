"""Dev-only loader for A1's two forbidden fields, `team` and `customers`.

The only place outside questions.py that opens a raw product file. It returns those two keys for a dev product and
refuses every other product, so the held-out split can never reach A1. The arms never see the raw file.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.questions import DEV_PRODUCTS, RAW_DIR


def load_fields(product: str, raw_dir: Path = RAW_DIR) -> dict:
    if product not in DEV_PRODUCTS:
        raise ValueError(f"A1 reads team and customers for dev products only; {product!r} is not a dev product")
    data = json.loads((Path(raw_dir) / "products" / f"{product}.json").read_text(encoding="utf-8"))
    return {"team": data.get("team", []), "customers": data.get("customers", [])}
