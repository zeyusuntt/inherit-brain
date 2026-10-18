"""agent_graph connects only as the agent user: no default-user connection to read host and port."""

import json

from inherit.config import load_settings
from inherit.tools import depth1


def test_agent_graph_parses_the_url_and_connects_only_as_the_agent_user(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "acl.json").write_text(json.dumps({"6.0": {"agent_Foo": "SENTINEL-PW"}}))
    seen = {}

    def no_from_url(*a, **k):
        raise AssertionError("FalkorDB.from_url opens a default-user connection")

    def fake_ro(host, port, user, password, name):
        seen["args"] = (host, port, user, password, name)
        return "graph"

    monkeypatch.setattr(depth1.FalkorDB, "from_url", no_from_url)
    monkeypatch.setattr(depth1, "ROOT", tmp_path)
    monkeypatch.setattr(depth1, "read_only_graph", fake_ro)
    monkeypatch.setattr(depth1, "load_settings",
                        lambda: type("S", (), {"falkor_urls": {"6.0": "redis://dbhost:6390"}})())
    assert depth1.agent_graph("Foo") == "graph"
    assert seen["args"] == ("dbhost", 6390, "agent_Foo", "SENTINEL-PW", "herb:Foo")

    monkeypatch.setattr(depth1, "load_settings", lambda: type("S", (), {"falkor_urls": {"6.0": "redis://h2"}})())
    depth1.agent_graph("Foo")
    assert seen["args"][:2] == ("h2", 6379)
