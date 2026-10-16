"""Settings loaded from .env. The API key is never printed or logged."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Hard project ceiling. NEBIUS_BUDGET_USD may lower the cap, never raise it.
PROJECT_CAP_USD = 50.0

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

# Integer pico-USD per token (1 USD = 10**12 pico-USD): (input, output).
# Chat model: Nebius list price 0.10 / 0.30 USD per 1M tokens.
# Embedding model: read from GET /models?verbose=true on 2026-10-15,
# pricing.prompt = 0.00000001 USD per token (0.01 USD per 1M), completion 0.
PRICES_PICO: dict[str, tuple[int, int]] = {
    MODEL: (100_000, 300_000),
    EMBED_MODEL: (10_000, 0),
}


@dataclass(frozen=True)
class Settings:
    nebius_base_url: str
    budget_usd: float
    proxy_port: int = 8787
    default_max_tokens: int = 1024
    model: str = MODEL
    embed_model: str = EMBED_MODEL
    meter_path: Path = ROOT / "runs" / "meter.jsonl"
    falkor_urls: dict[str, str] = field(
        default_factory=lambda: {"4.22": "redis://localhost:6379", "6.0": "redis://localhost:6380"}
    )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    budget = float(os.environ["NEBIUS_BUDGET_USD"])
    if not math.isfinite(budget) or budget < 0 or budget > PROJECT_CAP_USD:
        raise ValueError(f"NEBIUS_BUDGET_USD must be between 0 and {PROJECT_CAP_USD}")
    return Settings(
        nebius_base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        budget_usd=budget,
        proxy_port=int(os.environ.get("PROXY_PORT", "8787")),
    )


def load_upstream_secret() -> str:
    """The upstream provider key. Only inherit.proxy may call this."""
    load_dotenv(ROOT / ".env")
    return os.environ["NEBIUS_API_KEY"]
