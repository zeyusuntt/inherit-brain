"""One embedding helper for the loader and the tools: same model, 1024 dims, L2-normalised.

The API is asked for `dimensions=1024`. If it explicitly refuses the parameter, the first 1024 components are kept and re-normalised (valid for this Matryoshka-trained model). `STATE["path"]`
records which path ran. Corpus vectors are cached on disk by text hash, so a reload costs nothing.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import openai

from inherit import llm
from inherit.config import EMBED_MODEL, ROOT

DIM = 1024
ARM = "embed"
BATCH_ITEMS = 64
BATCH_CHARS = 100_000
QUERY_TASK = "Given a question about a company, retrieve the messages and documents that answer it"

STATE: dict = {"path": None, "cost_usd": 0.0, "prompt_tokens": 0, "calls": 0, "cache_hits": 0}


def _normalise(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)[:DIM]
    n = math.sqrt(float(a @ a))
    if not math.isfinite(n) or n == 0:
        raise ValueError("embedding has zero or non-finite norm")
    return (a / n).astype(np.float32)


UNSUPPORTED_PHRASES = ("not support", "unsupported", "unrecognized", "unknown parameter", "unexpected keyword",
                       "extra inputs", "not permitted", "not allowed")


def _unsupported(e: openai.BadRequestError) -> bool:
    """True only for an explicit refusal of the `dimensions` parameter itself."""
    msg = str(e).lower()
    return "dimensions" in msg and any(p in msg for p in UNSUPPORTED_PHRASES)


def _embed_batch(texts: list[str], embed_fn) -> list[np.ndarray]:
    try:
        res = embed_fn(texts, arm=ARM, dimensions=DIM)
        path = "api"
    except openai.BadRequestError as e:
        if not _unsupported(e):
            raise
        res = embed_fn(texts, arm=ARM)
        path = "truncate"
    if len(res.vectors) != len(texts):
        raise RuntimeError(f"embedding API returned {len(res.vectors)} vectors for {len(texts)} texts")
    if path == "api" and any(len(v) != DIM for v in res.vectors):
        raise RuntimeError(f"embedding API ignored dimensions={DIM} and returned {len(res.vectors[0])} components")
    if any(len(v) < DIM for v in res.vectors):
        raise RuntimeError(f"embedding shorter than {DIM}")
    STATE["path"] = path
    STATE["cost_usd"] += res.cost_usd
    STATE["prompt_tokens"] += res.prompt_tokens
    STATE["calls"] += 1
    return [_normalise(v) for v in res.vectors]


def _cache_file(cache_dir: Path, text: str) -> Path:
    return cache_dir / EMBED_MODEL / str(DIM) / (hashlib.sha256(text.encode("utf-8")).hexdigest() + ".npy")


def encode_corpus(texts: list[str], *, cache_dir: Path | None = None, embed_fn=None) -> list[list[float]]:
    if any(not t or not t.strip() for t in texts):
        raise ValueError("cannot embed blank text")
    embed_fn = embed_fn or llm.embed
    cache_dir = Path(cache_dir) if cache_dir else ROOT / "data" / "cache" / "embed"
    out: dict[str, np.ndarray] = {}
    todo: list[str] = []
    for t in dict.fromkeys(texts):
        f = _cache_file(cache_dir, t)
        if f.exists():
            out[t] = np.load(f)
            STATE["cache_hits"] += 1
        else:
            todo.append(t)
    batch: list[str] = []
    chars = 0
    for t in [*todo, None]:
        if t is not None and (len(batch) < BATCH_ITEMS and chars + len(t) <= BATCH_CHARS or not batch):
            batch.append(t)
            chars += len(t)
            continue
        if batch:
            for text, v in zip(batch, _embed_batch(batch, embed_fn)):
                f = _cache_file(cache_dir, text)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, v)
                out[text] = v
        batch, chars = ([t], len(t)) if t is not None else ([], 0)
    return [out[t].tolist() for t in texts]


def encode_query(text: str, *, embed_fn=None) -> list[float]:
    if not text or not text.strip():
        raise ValueError("cannot embed a blank query")
    prompt = f"Instruct: {QUERY_TASK}\nQuery:{text}"
    return _embed_batch([prompt], embed_fn or llm.embed)[0].tolist()
