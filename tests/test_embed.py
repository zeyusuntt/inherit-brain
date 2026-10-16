import math
from types import SimpleNamespace

import httpx
import openai
import pytest

from inherit import embed, llm
from inherit.llm import EmbedResult


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class Fake:
    """Stands in for llm.embed. `mode`: api (honours dimensions), reject (400 on dimensions), ignore (full 4096)."""

    def __init__(self, mode="api"):
        self.mode, self.calls = mode, []

    def __call__(self, texts, *, arm, dimensions=None, **_):
        self.calls.append((list(texts), arm, dimensions))
        if dimensions is not None and self.mode == "reject":
            resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
            raise openai.BadRequestError("This model does not support the dimensions parameter", response=resp, body=None)
        dim = dimensions if (dimensions and self.mode == "api") else 4096
        vecs = [[float((len(t) + i) % 7) + 1.0 for i in range(dim)] for t in texts]
        return EmbedResult(vecs, prompt_tokens=10 * len(texts), cost_usd=0.001)


def test_llm_embed_forwards_dimensions(monkeypatch):
    seen = {}

    class Raw:
        headers = {"X-Inherit-Cost-USD": "0.5"}

        def parse(self):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])], usage=SimpleNamespace(prompt_tokens=3))

    def create(**kw):
        seen.update(kw)
        return Raw()

    client = SimpleNamespace(embeddings=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_client", lambda arm, base_url: client)
    llm.embed(["a"], arm="embed", dimensions=1024)
    assert seen["dimensions"] == 1024
    seen.clear()
    llm.embed(["a"], arm="embed")
    assert "dimensions" not in seen


def test_corpus_api_path_is_normalised_1024(tmp_path):
    f = Fake("api")
    out = embed.encode_corpus(["alpha", "beta beta"], cache_dir=tmp_path, embed_fn=f)
    assert [len(v) for v in out] == [1024, 1024]
    assert all(abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-5 for v in out)
    assert f.calls[0][1:] == ("embed", 1024)
    assert embed.STATE["path"] == "api"


def test_corpus_truncates_when_dimensions_unsupported(tmp_path):
    f = Fake("reject")
    out = embed.encode_corpus(["alpha"], cache_dir=tmp_path, embed_fn=f)
    assert len(out[0]) == 1024
    assert abs(math.sqrt(sum(x * x for x in out[0])) - 1) < 1e-5
    full = [float((5 + i) % 7) + 1.0 for i in range(1024)]
    assert out[0] == pytest.approx(unit(full), rel=1e-5)
    assert embed.STATE["path"] == "truncate"


def test_silently_ignored_dimensions_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="ignored dimensions"):
        embed.encode_corpus(["alpha"], cache_dir=tmp_path, embed_fn=Fake("ignore"))


def test_unrelated_400_mentioning_dimension_propagates(tmp_path):
    def bad(texts, *, arm, dimensions=None):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        raise openai.BadRequestError("input exceeds the maximum dimension of context", response=resp, body=None)

    with pytest.raises(openai.BadRequestError):
        embed.encode_corpus(["a"], cache_dir=tmp_path, embed_fn=bad)


def test_other_errors_are_not_swallowed(tmp_path):
    def boom(texts, *, arm, dimensions=None):
        resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
        raise openai.InternalServerError("server error", response=resp, body=None)

    with pytest.raises(openai.InternalServerError):
        embed.encode_corpus(["a"], cache_dir=tmp_path, embed_fn=boom)


def test_cache_reuse_costs_nothing(tmp_path):
    f = Fake("api")
    a = embed.encode_corpus(["one", "two"], cache_dir=tmp_path, embed_fn=f)
    f2 = Fake("api")
    b = embed.encode_corpus(["two", "one", "three"], cache_dir=tmp_path, embed_fn=f2)
    assert f2.calls == [(["three"], "embed", 1024)]
    assert b[0] == pytest.approx(a[1]) and b[1] == pytest.approx(a[0])
    f3 = Fake("api")
    embed.encode_corpus(["one", "two", "three"], cache_dir=tmp_path, embed_fn=f3)
    assert f3.calls == []


def test_blank_text_is_refused(tmp_path):
    with pytest.raises(ValueError):
        embed.encode_corpus(["ok", "  "], cache_dir=tmp_path, embed_fn=Fake())


def test_query_uses_instruction_format():
    f = Fake("api")
    v = embed.encode_query("who fixed the login bug", embed_fn=f)
    sent = f.calls[0][0][0]
    assert sent.startswith("Instruct: ") and sent.endswith("\nQuery:who fixed the login bug")
    assert len(v) == 1024
