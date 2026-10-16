"""Per-product ACL users. Rules exactly as measured in docs/falkor-versions.md (the `%R~` form does not work).

`reset` first, so re-running removes every earlier grant. The default user is left alone locally; a public
deployment must lock it (Task 16).
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from inherit.config import ROOT

ACL_FILE = ROOT / "data" / "acl.json"


def agent_rules(p: str) -> list[str]:
    return [f"(~herb:{p} +GRAPH.RO_QUERY)"]


def whatif_rules(p: str) -> list[str]:
    return [f"(~whatif:{p}:* ~herb:{p} +GRAPH.COPY)",
            f"(~whatif:{p}:* +GRAPH.QUERY +GRAPH.RO_QUERY +GRAPH.DELETE)",
            f"(~herb:{p} +GRAPH.RO_QUERY)"]


def build_rules(p: str) -> list[str]:
    return [f"~herb:{p}", "+GRAPH.QUERY", "+GRAPH.RO_QUERY", "+GRAPH.DELETE", "+GRAPH.CONSTRAINT", "+INFO"]  # the client reads INFO on connect


def set_user(admin, user: str, password: str, rules: list[str]) -> None:
    admin.connection.execute_command("ACL", "SETUSER", user, "reset", "on", f">{password}", *rules)


def del_user(admin, user: str) -> None:
    admin.connection.execute_command("ACL", "DELUSER", user)


def drop_product_users(admin, product: str) -> None:
    """Delete agent_, whatif_ users and any whatif copies; call before dropping herb:<product>."""
    for u in (f"agent_{product}", f"whatif_{product}"):
        del_user(admin, u)
    for g in admin.list_graphs():
        if g.startswith(f"whatif:{product}:"):
            admin.select_graph(g).delete()


def setup_acl(admin, products: list[str], *, path: Path | None = None, label: str = "local") -> dict[str, str]:
    """Create agent_<p> and whatif_<p> with fresh random passwords; merge them into data/acl.json (mode 600)
    under the server `label`, since each server has its own passwords."""
    existing = set(admin.list_graphs())
    for p in products:
        if f"herb:{p}" not in existing:
            raise RuntimeError(f"herb:{p} must exist before its users are created")
    pw: dict[str, str] = {}
    for p in products:
        pw[f"agent_{p}"] = secrets.token_urlsafe(24)
        set_user(admin, f"agent_{p}", pw[f"agent_{p}"], agent_rules(p))
        pw[f"whatif_{p}"] = secrets.token_urlsafe(24)
        set_user(admin, f"whatif_{p}", pw[f"whatif_{p}"], whatif_rules(p))
    path = Path(path) if path else ACL_FILE
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved.setdefault(label, {}).update(pw)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(saved, f, indent=1, sort_keys=True)
    os.chmod(path, 0o600)
    return pw
