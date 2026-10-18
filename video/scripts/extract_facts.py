"""Write the facts the video shows, read from the demo site's data and the pre-registration.

Nothing on screen that is a fact is typed by hand: scores, intervals, token medians, the hash, the
audit rounds, the record excerpts, the Cypher and the what-if items all come from
`site/data/*.json` and `PREREG.md`, read only, into `video/src/content/facts.json`.
The plain-English arm names follow the video brief; each is checked against the arm's description.

Usage, from the repository root:
    python3 video/scripts/extract_facts.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site/data"
PREREG = ROOT / "PREREG.md"
RESULTS = ROOT / "results/RESULTS.md"
OUT = ROOT / "video/src/content/facts.json"

# Arm names on screen, in the run's arm order, with a word each description must contain.
ARM_NAMES = {
    "A5": ("inherit-brain", "company brain"),
    "A0": ("HERB's retrieval tools, same loop", "retrieval agent"),
    "A1f": ("The whole product in one prompt", "whole product"),
    "A0v": ("HERB's hybrid RAG", "hybrid RAG"),
    "A4": ("Depth-1 cards", "depth-1"),
    "A4c": ("Cards with a chain", "multi-hop answers step by step"),
    "A0f": ("Retrieval over cards", "Retrieval over the pre-extracted entity cards"),
    "A2": ("FalkorDB GraphRAG-SDK", "GraphRAG-SDK"),
}
COLD_OPEN_KINDS = [("Slack message", "Slack"), ("Meeting segment", "Meetings"), ("Document", "Documents"), ("PR", "Pull requests")]
PER_KIND = 16


def no_dashes(text: str) -> str:
    """On-screen text carries no em or en dashes (the site shows record dashes as hyphens too)."""
    return re.sub(r"\s+", " ", text.replace("—", "-").replace("–", "-")).strip()


def short(text: str, limit: int) -> str:
    text = no_dashes(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "..."


def stamp(ts: str) -> str:
    return ts.replace("T", " ")[:16]


def cold_open(product: dict) -> dict:
    evidence = list(product["evidence"].values())
    columns = []
    for kind, label in COLD_OPEN_KINDS:
        records = sorted((e for e in evidence if e.get("kind") == kind), key=lambda e: e["timestamp"])
        # Spread the picks over the whole span of months rather than the first dozen.
        step = max(1, len(records) // PER_KIND)
        picks = records[::step][:PER_KIND]
        rows = []
        for e in picks:
            where = e.get("title") or ""
            if kind == "Document":
                where = where.split(" (")[0]
            elif kind == "Meeting segment":
                where = re.sub(r",? line \d+$", "", where)
            rows.append({"id": e["id"], "where": short(where, 48), "author": no_dashes(e.get("author") or ""),
                         "ts": stamp(e["timestamp"]), "text": short(e.get("text") or "", 120)})
        columns.append({"kind": kind, "label": label, "records": rows})
    times = sorted(e["timestamp"] for e in evidence)
    return {"product": product["product"], "first": stamp(times[0])[:10], "last": stamp(times[-1])[:10], "columns": columns}


def question(product: dict, key: str) -> dict:
    return next(q for q in product["questions"] if q["k"] == key)


def walk_graph(product: dict) -> dict:
    q = question(product, "SupportForce/ans/3")
    a5 = q["arms"]["A5"]
    step = q["a5"]["path"][0]
    gold = set(q["gold"]["ids"])
    answered = list(zip(a5["answer_ids"], a5["answer_names"]))
    alice = [eid for eid, name in answered if name == "Alice Davis"]
    assert len(alice) == 2 and alice[0] != alice[1], alice
    return {
        "key": q["k"], "text": no_dashes(q["text"]), "tool": step["tool"], "params": step["params"],
        "rows": step["rows"], "ms": step["ms"], "n_queries": len(step["queries"]),
        "queries": [{"command": x["command"], "cypher": x["cypher"], "rows": x["rows"]} for x in step["queries"]],
        "answer": [{"eid": eid, "name": name, "in_gold": eid in gold} for eid, name in answered],
        "n_gold": len(gold), "n_correct": sum(eid in gold for eid, _ in answered), "score": a5["score"],
        "alice_davis": alice,
    }


def bug_board() -> dict:
    product = json.loads((SITE / "q/SecurityForce.json").read_text())
    q = question(product, "SecurityForce/ans/21")
    step = next(s for s in q["a5"]["path"] if s["tool"] == "bug_board")
    query = next(x for x in step["queries"] if ":FIXES]" in x["cypher"] and ":HAS_EVENT]" in x["cypher"])
    cypher = query["cypher"]
    # One clause per line, as the explorer's formatter does; the text itself is unchanged.
    lines = re.split(r" (?=MATCH |RETURN )", cypher)
    first = lines[0].split(" AND (")
    lines = [first[0]] + ["  AND (" + part for part in first[1:]] + lines[1:]
    assert "".join(lines).replace("  AND (", " AND (").replace(" ", "") == cypher.replace(" ", "")
    return {"key": q["k"], "tool": step["tool"], "command": query["command"], "rows": query["rows"], "ms": query["ms"], "lines": lines}


def whatif() -> dict:
    data = json.loads((SITE / "whatif.json").read_text())
    person = next(x for x in data["products"]["SupportForce"]["owners"] if x["person"]["name"] == "Julia Smith")
    items = [{"kind": i["kind"], "title": no_dashes(i["title"]), "id": i["id"]} for i in person["newly_unowned"]]
    return {"product": "SupportForce", "person": person["person"], "counts": person["counts"], "items": items,
            "bugs": sum(i["kind"] == "Bug" for i in items), "prs": sum(i["kind"] == "PR" for i in items)}


def prereg_section(number: int) -> str:
    text = PREREG.read_text(encoding="utf-8")
    start = re.search(rf"^## {number}\. ", text, re.M).start()
    end = re.search(rf"^## {number + 1}\. ", text, re.M).start()
    return text[start:end]


def gate(text: str) -> dict:
    m = re.match(r"^(.*?) (\d+/\d+)(?: \((.*)\))?$", text.strip())
    if not m:
        return {"label": text.strip(), "value": None, "note": None}
    return {"label": m.group(1), "value": m.group(2), "note": m.group(3)}


def audit_rounds() -> list[dict]:
    section = prereg_section(12)
    rows = []
    for line in section.splitlines():
        m = re.match(r"^\| (\d) \| (v\d) \| (.*?) \| (.*?) \|$", line)
        if m:
            failed = [] if m.group(3).strip() == "none" else [gate(g) for g in m.group(3).split(", ")]
            rows.append({"round": int(m.group(1)), "rubric": m.group(2), "failed": failed, "passed_text": no_dashes(m.group(4)),
                         "result": "passed" if not failed else "failed"})
    assert [r["round"] for r in rows] == [1, 2, 3, 4, 5, 6], rows
    assert [r["result"] for r in rows] == ["failed"] * 5 + ["passed"], rows
    return rows


def final_gates() -> list[dict]:
    """The round-6 gate table in PREREG section 12: audit gates with their numerators and denominators."""
    names = {"bug_event": "bug events", "review_relation": "review relations", "name_resolution": "names",
             "whatif_unowned": "newly unowned", "whatif_edge": "dependency edges", "recall_bug": "bug recall",
             "recall_reviewer": "reviewer recall"}
    out = []
    for line in prereg_section(12).splitlines():
        m = re.match(r"^\| audit:(?:precision|recall):(\w+) \| (\d+) \| (\d+) \| [0-9.]+ \| pass \|$", line)
        if m and m.group(1) in names:
            out.append({"label": names[m.group(1)], "value": f"{m.group(2)}/{m.group(3)}"})
    assert len(out) == 7, out
    return out


def products(questions: list[dict]) -> dict:
    line = next(l for l in prereg_section(3).splitlines() if "Six products are the development set" in l)
    dev = [p.strip() for p in line.split("development set:")[1].split(". The other")[0].split(",")]
    held = sorted({q["p"] for q in questions})
    assert len(dev) == 6 and len(held) == 24 and not set(dev) & set(held), (dev, held)
    return {"development": dev, "held_out": held}


def main() -> None:
    overview = json.loads((SITE / "overview.json").read_text())
    questions = json.loads((SITE / "questions.json").read_text())
    support = json.loads((SITE / "q/SupportForce.json").read_text())
    arms = overview["arms"]
    h1 = next(h for h in overview["hypotheses"] if h["id"] == "H1")
    seen = overview["also_reported"]["by_seen"]
    a5_tokens, a1f_tokens = arms["A5"]["median_input_tokens"], arms["A1f"]["median_input_tokens"]
    results_md = RESULTS.read_text(encoding="utf-8")
    # The overview and RESULTS.md must agree on the numbers the video says out loud.
    for needle in [f"+{h1['diff']} | [{h1['lo']}, {h1['hi']}]", f"{int(a5_tokens):,} / {int(a1f_tokens):,}",
                   f"+{seen['unseen']['H1']['diff']} [{seen['unseen']['H1']['lo']}, {seen['unseen']['H1']['hi']}]",
                   f"+{seen['seen']['H1']['diff']} [{seen['seen']['H1']['lo']}, {seen['seen']['H1']['hi']}]"]:
        if needle not in results_md:
            sys.exit(f"RESULTS.md does not contain {needle!r}")
    for arm, (_, word) in ARM_NAMES.items():
        if word.lower() not in arms[arm]["description"].lower():
            sys.exit(f"{arm}: description does not mention {word!r}")
    approved = overview["approved_utc"]
    facts = {
        "source": "video/scripts/extract_facts.py from site/data, PREREG.md and results/RESULTS.md; do not edit by hand",
        "n_questions": overview["n_questions"], "n_products_held_out": overview["n_products"],
        "products": products(questions),
        "prereg": {"sha256": overview["prereg_sha256"], "short": overview["prereg_sha256"][:8],
                   "posted_utc": approved.replace("T", " ")[:16] + " UTC", "gist_url": overview["gist_url"],
                   "wayback_url": overview["wayback_url"],
                   "commit": re.search(r"committed at `([0-9a-f]{7,40})`", results_md).group(1)},
        "model": "Qwen3-30B-A3B on Nebius",
        "model_id": re.search(r"`(Qwen/[^`]+)` on Nebius", PREREG.read_text(encoding="utf-8")).group(1),
        "arms": [{"id": a, "name": ARM_NAMES[a][0], "n": arms[a]["n"], "primary": arms[a]["primary"]} for a in overview["arm_order"]],
        "scores": {a: arms[a]["primary"] for a in ("A5", "A0", "A1f")},
        "tokens": {"A5": int(a5_tokens), "A1f": int(a1f_tokens), "ratio": int(a1f_tokens // a5_tokens)},
        "h1": {"diff": h1["diff"], "lo": h1["lo"], "hi": h1["hi"]},
        "by_seen": {k: {"n": v["n_questions"], **v["H1"]} for k, v in seen.items()},
        "audit": audit_rounds(),
        "audit_final": final_gates(),
        "cold_open": cold_open(support),
        "walk_graph": walk_graph(support),
        "bug_board": bug_board(),
        "whatif": whatif(),
        "site_url": "https://zeyusuntt.github.io/inherit-brain/",
        "site_label": "zeyusuntt.github.io/inherit-brain",
        "repo_label": "github.com/zeyusuntt/inherit-brain",
    }
    text = json.dumps(facts, indent=1, ensure_ascii=False)
    if "—" in text or "–" in text:
        sys.exit("facts.json would carry an em or en dash")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(facts['arms'])} arms, {len(facts['audit'])} audit rounds, "
          f"{sum(len(c['records']) for c in facts['cold_open']['columns'])} cold-open records")


if __name__ == "__main__":
    main()
