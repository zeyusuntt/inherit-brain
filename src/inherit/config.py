"""Settings loaded from .env. The API key is never printed or logged."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str = field(repr=False)
    nebius_base_url: str
    budget_usd: float
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # USD per token. Nebius list price 0.10 / 0.30 USD per 1M tokens.
    price_in: float = 0.10 / 1e6
    price_out: float = 0.30 / 1e6
    # Read from GET /models?verbose=true on 2026-10-15: pricing.prompt = 0.00000001
    # USD per token (0.01 USD per 1M), completion 0.
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    embed_price_in: float = 0.01 / 1e6
    meter_path: Path = ROOT / "runs" / "meter.jsonl"
    falkor_urls: dict[str, str] = field(
        default_factory=lambda: {"4.22": "redis://localhost:6379", "6.0": "redis://localhost:6380"}
    )


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        nebius_api_key=os.environ["NEBIUS_API_KEY"],
        nebius_base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        budget_usd=float(os.environ["NEBIUS_BUDGET_USD"]),
    )
