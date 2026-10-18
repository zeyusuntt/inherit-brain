"""Build the static JSON for the demo site from the finished held-out run.

    uv run python site/build_data.py

Reads results/, runs/eval/<ARM>-held (records, grades, tool envelopes), the bundle (data/bundle) and, read-only,
the 6.0 FalkorDB graphs (port 6380). Makes no model call, runs no arm and writes only under site/data/.
Held-out questions are loaded only through eval.questions.load_questions("held") (project rule 2).
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import grade as G  # noqa: E402
from eval import run as R  # noqa: E402
from eval.questions import load_questions  # noqa: E402
from inherit.data.bundle import load_metadata  # noqa: E402
from inherit.data.records import load_records  # noqa: E402

OUT = ROOT / "site" / "data"
BUNDLE = ROOT / "data" / "bundle"
RUNS = ROOT / "runs" / "eval"
ARMS = ("A5", "A0", "A1f", "A0v", "A4", "A4c", "A0f", "A2")  # order of the results table
TEXT_MAX = 600  # characters of a record excerpt
ANSWER_MAX = 1500  # characters of a stored answer
MAX_IDS = 50
MAX_TOOL_EVIDENCE = 25  # extra record ids per question taken from tool rows (final evidence ids come first)
MAX_LEADS = 3
MAX_OWNERS = 5
TOTAL_SPEND_USD = 42.72

ARM_DESCRIPTIONS = {
    "A5": "The company brain: nine read-only tools on the FalkorDB graph (multi-hop queries, aggregation across "
          "cards, algorithms), driven by the shared agent loop. The 'if this person leaves' counterfactual is a "
          "separate feature and was not part of the scored run.",
    "A0": "The HERB paper's retrieval agent over the raw records, run in the shared loop with the same limits (12 "
          "tool calls, 60,000 input tokens). The comparator of H1: same model, no graph.",
    "A1f": "Reading the whole product: one model call whose context holds all of the product's records "
           "(about 110,000 to 140,000 tokens). The comparator of H2.",
    "A0v": "The HERB paper's hybrid RAG with a single retrieval (BM25 top 20, then vector top 20) and one model "
           "call over those chunks, no tools, no loop.",
    "A4": "Typed depth-1 access to the entity cards (resolve, get_card, neighbors), with exact aggregation; the "
          "restricted card view. Five products plus the multi-hop questions.",
    "A4c": "The same cards and depth-1 tools, with the model assembling multi-hop answers step by step. Five products.",
    "A0f": "Retrieval over the pre-extracted entity cards with the shared loop: the A0 agent on the card "
           "representation. Five products.",
    "A2": "GraphRAG-SDK 1.4.0 with its automatic ontology, one model call over its retrieved context. Six sampled "
          "products.",
}
READING = [
    "A0f minus A0: the effect of the pre-extracted representation on a retrieval agent (representation and "
    "retrieved content change together).",
    "A4 minus A0f: the same cards, typed access and exact aggregation against retrieval.",
    "A5 minus A4: the full multi-hop toolkit (multi-hop queries, aggregation across cards, algorithms) against the "
    "restricted card view; not attributed to multi-hop alone.",
    "A5 against A4c: multi-hop inside fixed tools against the model assembling it step by step (accuracy, calls, "
    "tokens, latency).",
    "Every difference is measured under the registered resource limits. An arm that stops on its budget more often "
    "is not given more; budget-stop rates are reported per arm.",
    "None of these differences shows that another database could not do the same.",
]


def r1(x):
    return None if x is None else round(float(x), 1)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def cut(s, n):
    s = s if isinstance(s, str) else ("" if s is None else str(s))
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------- overview
def budget_stops() -> dict[str, str | None]:
    """Budget-stop column of the per-arm table in results/RESULTS.md ("n/a" becomes None)."""
    out: dict[str, str | None] = {}
    for line in (ROOT / "results" / "RESULTS.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 15 and cells[0] in ARMS:
            out[cells[0]] = None if cells[13] == "n/a" else cells[13]
    return out


def _ci(b: dict) -> dict:
    return {"diff": r1(b["diff"]), "lo": r1(b["lo"]), "hi": r1(b["hi"])}


def also_reported(res: dict) -> dict:
    """PREREG section 10's 'also reported' items: H1 and H2 by seen and unseen question strings, with the per-arm
    primary endpoint of each half, and the overlap of cited records with HERB's citations."""
    ex = json.loads((ROOT / "results" / "reported_extras.json").read_text(encoding="utf-8"))
    by_seen = {}
    for split in ("seen", "unseen"):
        b = ex["by_seen"][split]
        by_seen[split] = {"n_questions": b["n_questions"], "H1": _ci(b["H1"]), "H2": _ci(b["H2"]),
                          "primary": {a: r1(res["summary"][a]["by_seen"][split]["primary"])
                                      for a in ("A5", "A0", "A1f", "A0v")}}
    cites = {a: {"n": v["n"], "precision": r1(100 * v["mean_precision"]), "recall": r1(100 * v["mean_recall"]),
                 "no_evidence": v["n_empty_evidence"]} for a, v in ex["citation_overlap"].items()}
    return {"by_seen": by_seen, "citation_overlap": cites}


def build_overview() -> dict:
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    rel = json.loads((ROOT / "results" / "freeze" / "release.json").read_text(encoding="utf-8"))
    stops = budget_stops()
    arms = {}
    for a in ARMS:
        s = res["summary"][a]
        arms[a] = {
            "description": ARM_DESCRIPTIONS[a],
            "n": s["n"]["answerable"] + s["n"]["unanswerable"],
            "n_answerable": s["n"]["answerable"], "n_unanswerable": s["n"]["unanswerable"],
            "primary": r1(s["primary"]), "paper_style": r1(s["paper_macro"]),
            "per_type": {t: {"n": v["n"], "mean": r1(v["mean"]), "failed": v["failed"]}
                         for t, v in s["per_type"].items()},
            "unanswerable_correct_refusal_judge": r1(s["unanswerable_rate"]),
            "explicit_refusal_rate": r1(s["explicit_refusal_rate"]),
            "false_refusal_rate": r1(s["false_refusal_rate"]),
            "failed": s["failed"]["answerable"] + s["failed"]["unanswerable"],
            "failure_rate": r1(s["failure_rate"]),
            "median_input_tokens": s["median_input_tokens"], "mean_tool_calls": r1(s["mean_tool_calls"]),
            "budget_stop_rate": stops.get(a),
            "latency_p50_s": r1(s["latency_p50"]), "latency_p95_s": r1(s["latency_p95"]),
        }

    def ci(b):
        return {"diff": r1(b["diff"]), "lo": r1(b["lo"]), "hi": r1(b["hi"])}

    b, d = res["boot"], res["decisions"]
    hyp = [
        {"id": "H1", "comparison": "A5 minus A0, primary endpoint", **ci(b["H1"]),
         "threshold": "lower bound > 0 (prior: at least 15)", "result": "holds" if d["H1"]["pass"] else "fails"},
        {"id": "H2", "comparison": "A5 minus A1f, primary endpoint", **ci(b["H2"]),
         "threshold": "lower bound > -3", "result": "holds" if d["H2"]["interval_ok"] else "fails"},
        {"id": "H2 tokens", "comparison": "A5 median input tokens / A1f's", "diff": None, "lo": None, "hi": None,
         "value": f"{100 * d['H2']['token_ratio']:.1f}% ({arms['A5']['median_input_tokens']:,.0f} / "
                  f"{arms['A1f']['median_input_tokens']:,.0f})",
         "threshold": "at most 10%", "result": "holds" if d["H2"]["tokens_ok"] else "fails"},
        {"id": "H3", "comparison": "A5 minus A4, multi-hop family", **ci(b["H3"]),
         "threshold": "lower bound > 0 (prior: at least 10)",
         "result": ("holds" if d["H3"]["pass"] else "fails") + f", on {res['multihop_questions']} questions"},
        {"id": "H3 all", "comparison": "A5 minus A4 on every question A4 answered", **ci(b["H3_all"]),
         "threshold": "reported, no threshold", "result": ""},
        {"id": "H4", "comparison": "correct refusal rate (HERB judge)", "diff": None, "lo": None, "hi": None,
         "value": f"A5 {d['H4']['unanswerable_rate']['A5']:.1f}%, A1f {d['H4']['unanswerable_rate']['A1f']:.1f}%",
         "threshold": "A5 at least A1f", "result": "holds" if d["H4"]["pass"] else "fails"},
    ]
    secondary = [{"contrast": k, **ci(v)} for k, v in res["secondary"].items()]
    p502 = res["post_hoc_sensitivity_excluding_provider_502"]
    cust = res["post_hoc_sensitivity_company_cust_ids_mapped"]
    return {
        "title": "inherit-brain on HERB, held-out run",
        "attribution": "HERB (Salesforce AI Research), CC BY-NC 4.0",
        "n_questions": 1203, "n_products": 24, "bootstrap": "product-clustered paired bootstrap, 10,000 replicates",
        "outcome_sentence": "On Salesforce's HERB questions, 40.8 points above an agent using the same model, no "
                            "worse than 3 points below reading the whole product, with 8.0% of its tokens, and a "
                            "path behind every answer.",
        "prereg_sha256": rel["prereg_sha256"], "gist_url": rel["post_url"], "wayback_url": rel["wayback_url"],
        "approved_utc": rel["approved_utc"],
        "total_spend_usd": TOTAL_SPEND_USD, "spend_cap_usd": 50.0,
        "arm_order": list(ARMS), "arms": arms,
        "arm_note": "A4, A4c and A0f ran on five seeded held-out products (A4 also on the 7 multi-hop questions); A2 "
                    "on six sampled products. Their columns are not comparable to the full arms; the contrasts "
                    "pair them with A5 or A0 on the same questions.",
        "hypotheses": hyp,
        "paper_style": {"H1": ci(res["paper_style"]["H1"]), "H2": ci(res["paper_style"]["H2"])},
        "secondary_contrasts": secondary,
        "how_differences_are_read": READING,
        "post_hoc": [
            {"title": "Provider network errors",
             "text": "44 of A0's and 6 of A1f's failures were the proxy's 502 upstream unreachable errors from the "
                     "model provider, not the arms' own failures. They stay zero-scored in the main table, as "
                     "registered. Dropping those slots from both arms of the pair:",
             "H1": ci(p502["H1"]), "H2": ci(p502["H2"]),
             "dropped_slots": {"H1": p502["H1"]["dropped_slots"], "H2": p502["H2"]["dropped_slots"]}},
            {"title": "Company answers given as customer ids",
             "text": "All arms were told to answer a company by its name, never its CUST- id; A1f gave CUST- ids in "
                     "49 of its 104 company answers (A0 in 3, A5 in none), which score 0 under the registered id "
                     "matching. Mapping CUST- ids to company names through the customer directory, the same way "
                     "for every arm, and rescoring with the same F1:",
             "primary_endpoint": {k: r1(v) for k, v in cust["primary_endpoint"].items()},
             "changed_company_answers": cust["changed_company_answers"],
             "H1": ci(cust["H1"]), "H2": ci(cust["H2"])},
        ],
        "also_reported": also_reported(res),
        "caveats": [
            "The judge is model M, so content and unanswerable scores are not comparable with the HERB paper's "
            "GPT-4o-judged numbers.",
            "H3 rests on one multi-hop family with 7 held-out questions.",
            "The loop arms share a 60,000-token input budget; an arm that runs out of budget is not given more.",
            "Scores are F1 x 100 (ids) or the HERB judge on model M (content, unanswerable). Failed slots are scored 0 "
            "and kept in the denominator.",
        ],
    }


# ---------------------------------------------------------------- records (bundle only)
class Lookup:
    def __init__(self, product: str):
        r = load_records(BUNDLE, product)
        self.r = r
        self.msg = {m.id: m for m in r.messages}
        self.pr = {p.id: p for p in r.prs}
        self.seg = {s.id: s for m in r.meetings for s in m.segments}
        self.meeting = {m.id: m for m in r.meetings}
        self.doc = {d.id: d for d in r.documents}
        self.chat = {c.id: c for c in r.chats}
        self.url = {u.id: u for u in r.urls}

    def who(self, x) -> str | None:
        e = self.r.employees.get(x) if x else None
        return e["name"] if e else x

    def excerpt(self, rid: str) -> dict | None:
        ex = self._excerpt(rid.split(":")[2] if rid.startswith("BugEvent:") and rid.count(":") >= 3 else rid)
        if ex is not None:
            ex["id"] = rid  # a BugEvent id cites the message it was read from (BugEvent:<product>:<message id>:<n>)
        return ex

    def _excerpt(self, rid: str) -> dict | None:
        if rid in self.msg:
            m = self.msg[rid]
            return {"id": rid, "kind": "Slack message", "title": f"#{m.channel_name}", "author": self.who(m.user_id),
                    "timestamp": m.ts, "text": cut(m.text, TEXT_MAX)}
        if rid in self.seg:
            s = self.seg[rid]
            m = self.meeting[s.meeting_id]
            return {"id": rid, "kind": "Meeting segment", "title": f"{m.document_type} {m.id}, line {s.line_no}",
                    "author": s.speaker, "timestamp": m.date, "text": cut(s.text, TEXT_MAX)}
        if rid in self.doc:
            d = self.doc[rid]
            return {"id": rid, "kind": "Document", "title": f"{d.type} ({d.id})", "author": self.who(d.author),
                    "timestamp": d.date, "text": cut(d.content, TEXT_MAX)}
        if rid in self.pr:
            p = self.pr[rid]
            return {"id": rid, "kind": "PR", "title": p.title, "author": self.who(p.author), "timestamp": p.created_at,
                    "text": cut(f"{p.summary} (state {p.state}, merged {p.merged}) {p.link}", TEXT_MAX)}
        if rid in self.url:
            u = self.url[rid]
            return {"id": rid, "kind": "URL", "title": u.link, "author": None, "timestamp": None,
                    "text": cut(" / ".join(u.descriptions), TEXT_MAX)}
        if rid in self.chat:
            c = self.chat[rid]
            return {"id": rid, "kind": "Meeting chat", "title": c.meeting_id or rid, "author": None,
                    "timestamp": c.date, "text": cut(c.text, TEXT_MAX)}
        if rid in self.meeting:
            m = self.meeting[rid]
            return {"id": rid, "kind": "Meeting", "title": f"{m.document_type} {m.id}", "author": None,
                    "timestamp": m.date, "text": cut(" ".join(s.text for s in m.segments[:6]), TEXT_MAX)}
        return None


# ---------------------------------------------------------------- A5 tool path
def read_envelopes(run_dir: Path, slot: str):
    f = run_dir / "tools" / f"{slot.replace('/', '__')}.json.gz"
    if not f.exists():
        return [], None
    env = json.load(gzip.open(f, "rt", encoding="utf-8"))
    diag = next((e["agent"] for e in env if isinstance(e, dict) and "agent" in e), None)
    return [e for e in env if isinstance(e, dict) and "tool" in e], diag


def starters(all_envs: dict[str, list]) -> dict[str, set]:
    """Per tool, the first query text of calls whose op block is unambiguous (the next call is another tool)."""
    st: dict[str, set] = defaultdict(set)
    for envs, ops in all_envs.values():
        i = 0
        for k, e in enumerate(envs):
            if "error" in e:
                if i < len(ops) and ops[i].command == "refused":
                    i += 1
                continue
            j = i
            while j < len(ops) and ops[j].tool == e["tool"] and ops[j].command != "refused":
                j += 1
            nxt = envs[k + 1]["tool"] if k + 1 < len(envs) else None
            if nxt != e["tool"] and j > i:
                st[e["tool"]].add(ops[i].query)
            i = j
    return st


def split_calls(envs: list, ops: list, st: dict[str, set]):
    """Assign the run's flat op list to its tool calls, in order. Returns (per-call op lists, ambiguous flag).

    A call's ops are the contiguous ops of its tool; a refused call owns one 'refused' op. Where two calls of the
    same tool follow each other the block is cut at the next op that starts a call (a first query seen elsewhere
    to start that tool)."""
    out, i, ambiguous = [], 0, False
    for k, e in enumerate(envs):
        if "error" in e:
            if i < len(ops) and ops[i].command == "refused":
                out.append([ops[i]])
                i += 1
            else:
                out.append([])
            continue
        j = i
        again = k + 1 < len(envs) and envs[k + 1]["tool"] == e["tool"] and "error" not in envs[k + 1]
        while j < len(ops) and ops[j].tool == e["tool"] and ops[j].command != "refused":
            if again and j > i and ops[j].query in st.get(e["tool"], ()):
                break
            j += 1
        if again and (j == i or ops[i].query not in st.get(e["tool"], ())):
            ambiguous = True  # a call of a run of same-tool calls that does not start with a known first query
        out.append(ops[i:j])
        i = j
    return out, ambiguous or i != len(ops)


def row_preview(row: dict) -> dict:
    keep = {k: row[k] for k in ("answer", "name", "title", "kind", "role", "group", "via") if k in row}
    for k in ("answer", "name", "title"):
        if k in keep and isinstance(keep[k], str):
            keep[k] = cut(keep[k], 160)
    keep["refs"] = list(row.get("refs") or [])[:4]
    keep["evidence"] = list(row.get("evidence") or [])[:3]
    return keep


def tool_path(envs: list, ops: list, st: dict[str, set]) -> tuple[list[dict], bool]:
    groups, ambiguous = split_calls(envs, ops, st)
    calls = []
    for n, (e, g) in enumerate(zip(envs, groups), 1):
        rows = e.get("rows") or []
        c = {"n": n, "tool": e["tool"], "params": e.get("params"), "rows": len(rows),
             "ms": round(sum(o.ms for o in g), 1),
             "queries": [{"command": o.command, "cypher": o.query, "params": o.params, "rows": o.row_count,
                          "ms": round(o.ms, 1)} for o in g if o.command != "refused"]}
        if "error" in e:
            c["error"] = cut(e["error"], 300)
        else:
            c["truncated"] = bool(e.get("truncated"))
            c["top_rows"] = [row_preview(r) for r in rows[:5]]
            if e.get("note"):
                c["note"] = cut(e["note"], 300)
        calls.append(c)
    return calls, ambiguous


def tool_evidence_ids(envs: list) -> list[str]:
    seen: dict[str, None] = {}
    for e in envs:
        for r in e.get("rows") or []:
            for x in (r.get("evidence") or []):
                if isinstance(x, str):
                    seen.setdefault(x)
    return list(seen)


# ---------------------------------------------------------------- questions and per-product files
def names_for(ids, q_type, emp, cust):
    if q_type == "person":
        return [(emp.get(i) or {}).get("name") for i in ids]
    if q_type == "company":
        return [(cust.get(i) or {}).get("name") if i in cust else None for i in ids]
    return None


def build_questions_and_products() -> dict:
    questions = load_questions("held")
    emp, cust_list = load_metadata(BUNDLE)
    cust = {c["id"]: c for c in cust_list}
    cust_company = {k: {"name": v.get("company") or v.get("name")} for k, v in cust.items()}
    runs, graded = {}, {}
    for a in ARMS:
        rd = RUNS / f"{a}-held"
        runs[a] = R.read_records(rd)
        graded[a] = {str(g.key): g for g in G.load_graded(rd)}

    a5dir = RUNS / "A5-held"
    a5env = {s: read_envelopes(a5dir, s) for s in runs["A5"]}
    st = starters({s: (envs, runs["A5"][s].tool_ops) for s, (envs, _) in a5env.items()})

    index, by_product = [], defaultdict(list)
    ambiguous_slots = 0
    for q in questions:
        slot = str(q.key)
        scores = {a: (round(graded[a][slot].score, 1) if slot in graded[a] else None) for a in ARMS}
        index.append({"k": slot, "p": q.product, "t": q.type, "q": q.text, "seen": q.seen, "s": scores})
        by_product[q.product].append(q)

    prod_dir = OUT / "q"
    n_excerpts = 0
    sizes_unresolved = 0
    for product, qs in sorted(by_product.items()):
        look = Lookup(product)
        evidence: dict[str, dict] = {}

        def need(rid):
            nonlocal sizes_unresolved
            if rid in evidence:
                return
            ex = look.excerpt(rid)
            if ex is None:
                sizes_unresolved += 1
                ex = {"id": rid, "kind": None, "title": None, "author": None, "timestamp": None, "text": None}
            evidence[rid] = ex

        rows = []
        for q in qs:
            slot = str(q.key)
            if q.type in ("person", "company", "pr", "url"):
                ids = list(q.gold or [])
                gold = {"ids": ids, "names": names_for(ids, q.type, emp, cust_company)}
            elif q.type == "content":
                g = q.gold
                gold = {"text": cut(" ".join(g) if isinstance(g, list) else g, 2000)}
            else:
                gold = None
            arms = {}
            for a in ARMS:
                rec = runs[a].get(slot)
                if rec is None:
                    continue
                f, g = rec.final, graded[a][slot]
                ids = list(f.answer_ids)[:MAX_IDS]
                arms[a] = {"outcome": f.outcome, "answer_text": cut(f.answer_text, ANSWER_MAX), "answer_ids": ids,
                           "answer_names": names_for(ids, q.type, emp, cust_company),
                           "score": round(g.score, 1), "failed": g.failed}
            entry = {"k": slot, "type": q.type, "text": q.text, "seen": q.seen, "gold": gold, "arms": arms}
            rec = runs["A5"].get(slot)
            if rec is not None:
                envs, diag = a5env[slot]
                calls, amb = tool_path(envs, rec.tool_ops, st)
                ambiguous_slots += amb
                final_ev = list(rec.final.evidence_ids)[:10]
                extra = [x for x in tool_evidence_ids(envs) if x not in final_ev][:MAX_TOOL_EVIDENCE]
                for rid in final_ev + extra:
                    need(rid)
                u = rec.final.usage
                entry["a5"] = {
                    "path": calls, "ops_assignment_ambiguous": amb,
                    "evidence_ids": final_ev, "tool_row_evidence_ids": extra,
                    "diagnostics": {"model_calls": u.n_calls, "tool_calls": len(calls),
                                    "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                                    "cost_usd": round(u.cost_usd, 6), "latency_s": round(rec.latency_s, 2),
                                    "budget_stop": (diag or {}).get("budget_stop"),
                                    "model_outcome": (diag or {}).get("model_outcome")},
                }
            rows.append(entry)
        n_excerpts += len(evidence)
        dump(prod_dir / f"{product}.json", {"product": product, "questions": rows, "evidence": evidence})
    dump(OUT / "questions.json", index)
    return {"questions": len(index), "products": len(by_product), "excerpts": n_excerpts,
            "unresolved_evidence_ids": sizes_unresolved, "ambiguous_op_slots": ambiguous_slots}


# ---------------------------------------------------------------- what-if
def build_whatif(products: list[str]) -> dict:
    """Newly unowned and affected items for up to three Engineering Leads per product.

    The graph is read once with RO queries (the same export the independent check uses) and the three groups are
    recomputed by inherit.whatif_check.check, which results/freeze/whatif_check.json shows equal to the graph
    counterfactual on every lead (0 mismatches on 4.22 and 6.0). Nothing is copied or written on the server."""
    from inherit import whatif
    from inherit.whatif_check import DEPENDENCY_RULES, check, engineering_leads, work_owners
    from inherit.whatif_schema import MAX_HOPS, export_snapshot

    out = {}
    for product in products:
        snap = export_snapshot(whatif.whatif_connection(product, "6.0"), product)
        node = {n.id: n for n in snap.nodes}
        dep = {}  # (dependent, prerequisite) -> rule
        for e in snap.edges:
            for typ, sl, dl, need_, end in DEPENDENCY_RULES:
                if (e.type == typ and node[e.src].label == sl and node[e.dst].label == dl
                        and all(e.props.get(k) == v for k, v in need_.items())):
                    dep[(e.src, e.dst) if end == "src" else (e.dst, e.src)] = typ
        prereq = defaultdict(list)
        for (d, p) in sorted(dep):
            prereq[d].append(p)

        def item(i):
            n = node[i]
            return {"id": i, "kind": n.label, "title": n.props.get("title") or n.props.get("link") or i}

        def witness(a, targets):
            queue, prev = [a], {a: None}
            for _ in range(MAX_HOPS):
                nxt = []
                for x in queue:
                    for p in prereq[x]:
                        if p not in prev:
                            prev[p] = x
                            nxt.append(p)
                            if p in targets:
                                path, cur = [], p
                                while cur is not None:
                                    path.append(cur)
                                    cur = prev[cur]
                                return path[::-1]
                queue = nxt
            return [a]

        def entry(eid, res):
            p = node[f"Person:{eid}"].props
            paths = []
            for a in sorted(res.affected):
                path = witness(a, set(res.newly_unowned))
                paths.append({"affected": item(a), "path": [item(x) for x in path],
                              "rules": [dep.get((x, y)) for x, y in zip(path, path[1:])]})
            return {"person": {"eid": eid, "name": p.get("name"), "role": p.get("role")},
                    "counts": {"already_unowned": len(res.already_unowned),
                               "newly_unowned": len(res.newly_unowned), "affected": len(res.affected)},
                    "bound_hit": res.bound_hit, "max_hops": MAX_HOPS,
                    "newly_unowned": [item(i) for i in sorted(res.newly_unowned)],
                    "affected": paths}

        leads = [(eid, check(snap, eid)) for eid in engineering_leads(snap)]
        leads.sort(key=lambda t: (-len(t[1].newly_unowned), -len(t[1].affected), t[0]))
        rows = [entry(eid, res) for eid, res in leads[:MAX_LEADS]]
        owners = [(eid, check(snap, eid)) for eid in work_owners(snap)]
        owners.sort(key=lambda t: (-len(t[1].newly_unowned), -len(t[1].affected),
                                   node[f"Person:{t[0]}"].props.get("name") or "", t[0]))
        owner_rows = [entry(eid, res) for eid, res in owners[:MAX_OWNERS]]
        out[product] = {"leads_in_product": len(leads), "leads": rows, "owners": owner_rows}
    return {
        "basis": "Computed from the responsibility and dependency recorded in the data, not a prediction of real "
                 "consequences. Documents are not counted: the records do not say whether their feedback is "
                 "still open.",
        "owners_note": "The owners list is not limited to Engineering Leads. It takes everyone in the product who "
                       "opened a PR or was assigned a bug, runs the same calculation for each, and keeps the five "
                       "whose departure leaves the most open items without an owner. Ties go to the person with more "
                       "affected items, then to the name.",
        "definitions": "Open work is an active bug or an open, unmerged PR. A bug's owners are the people on its "
                       "latest ASSIGNED event; a PR's owner is its author. Newly unowned: open items whose only owner "
                       "is the person who leaves. Affected: open items that depend (up to 4 hops, through the "
                       "feature, PR and bug links in the records) on a newly unowned item.",
        "products": out,
    }


def main() -> int:
    held = sorted({q.product for q in load_questions("held")})
    dump(OUT / "overview.json", build_overview())
    stats = build_questions_and_products()
    dump(OUT / "whatif.json", build_whatif(held))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
