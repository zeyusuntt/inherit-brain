"""Throwaway diagnostic: the bounded oracle-plan score of the Task 8 tool catalogue on the six dev graphs.

    uv run python scripts/dev_oracle_plans.py [--server 6.0|4.22] [--misses]

One deterministic plan per question family (the frozen mapping of task-8-design-review.md finding 9). A plan reads only
the question text and the rows the tools returned; gold is used only by the evaluator at the bottom. Plans run with the
agent's budgets (12 calls, paging through cursors, 200 rows / 12,000 tokens) and are scored with the official ID scoring
(`eval.grade.f1_score_sets`); content questions are scored separately by key-phrase coverage. Each score is reported
before and after the output guard. Families whose facts do not exist yet (semantic layer, Task 4c) print `pending 4c`.
Prints IDs and scores only, never gold text. Plans and gold never enter prompts, facts or the catalogue.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from eval.grade import f1_score_sets, normalize
from eval.questions import load_questions
from inherit.extract.names import ROLE_NOUNS
from inherit.tools.catalog import ToolError
from inherit.tools.depth1 import agent_graph
from inherit.tools.guard import guard
from inherit.tools.trace import Trace

PRODUCTS = ("ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce")
ROLES = {r.lower(): r for r in ROLE_NOUNS}
PARTIAL = {"person: authors and key reviewers": "authors only until the semantic layer adds reviewers"}


class Plan:
    """One question's tool calls. `call` pages through cursors; every page is a call against the 12."""

    def __init__(self, g, product, text):
        self.g, self.product, self.text, self.trace, self.results = g, product, text, Trace(), []

    def call(self, tool, **params):
        rows = []
        while True:
            res = self.trace.run(self.g, self.product, tool, **params)
            self.results.append(res)
            rows += res.rows
            if not res.truncated:
                return rows
            params = {**params, "cursor": res.cursor}


def uniq(rows, key="answer"):
    return list(dict.fromkeys(r[key] for r in rows))


def role_in(text):
    m = re.search(r"(?:IDs? of|by) (?:the )?(.+?)s? (?:who|to improve)", text.replace("’", "'"))
    return ROLES.get(m.group(1).lower()) if m else None


def component_in(text, x):
    for pat in (rf"issues? (?:in|with) the (.+?) (?:of|in) {x}\b", rf"(?:affected by|resolved) the (.+?) issues? in {x}\b",
                rf"issues in the (.+?) (?:of|in) {x}\b"):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def feature_in(text, x):
    m = re.search(rf"PRs (?:for|related to) (?:the )?(.+?)(?: feature)? in {x}\b", text)
    return re.sub(r"^implementing ", "", m.group(1)) if m else None


def families():
    """(name, regex over the question text with <X> for the product, needs, plan). `needs`: facts that must exist."""
    X = r"(?P<x>\w+)"
    F: list = []

    def fam(name, pat, needs, fn, kind):
        F.append((name, re.compile(pat), needs, fn, kind))

    def company_max(status):
        return lambda p: ids(p.call("bug_board", status=status, actor="company", rank="max"))

    ids = uniq
    fam("company: max unresolved", r"maximum number of unresolved bugs", None, company_max("unresolved"), "ids")
    fam("company: max no-fix", r"maximum number of issues that didn.t need fixes", None, company_max("no_fix_needed"), "ids")
    fam("company: max resolved", r"maximum number of resolved bugs", None, company_max("resolved"), "ids")
    fam("company: max reopened", r"maximum number of reopened bugs", None, company_max("reopened"), "ids")
    fam("company: max active", r"maximum number of active bugs", None, company_max("active"), "ids")
    fam("company: previous release issues", r"companies that reported issues with the previous release", "Observation",
        lambda p: ids(p.call("who_said", subject="previous_release", kind="issue", **{"return": "companies"})), "ids")
    fam("company: previous release strengths", r"companies that highlighted strengths of the previous release", "Observation",
        lambda p: ids(p.call("who_said", subject="previous_release", kind="strength", **{"return": "companies"})), "ids")
    fam("company: affected by component issues", r"companies were affected by", None,
        lambda p: ids(p.call("bug_board", status="any", component=component_in(p.text, p.product), actor="company",
                             rank="all")), "ids")
    fam("person: resolved component issues", r"engineers? who resolved (?!the maximum)", None,
        lambda p: ids(p.call("bug_board", status="resolved", component=component_in(p.text, p.product), actor="resolver",
                             rank="all")), "ids")
    fam("person: resolved max bugs", r"resolved the maximum number of customer bugs", None,
        lambda p: ids(p.call("bug_board", status="resolved", actor="resolver", rank="max")), "ids")
    fam("person: assigned most unresolved", r"currently working on the highest number of unresolved", None,
        lambda p: ids(p.call("bug_board", status="unresolved", actor="assignee", rank="max")), "ids")
    fam("person: assigned most active", r"assigned to the highest number of active bugs", None,
        lambda p: ids(p.call("bug_board", status="active", actor="assignee", rank="max")), "ids")
    fam("person: fix PRs of reopened bugs", r"authored maximum number of PRs .* now reopened", None,
        lambda p: ids(p.call("bug_board", status="reopened", actor="fix_pr_author", rank="max")), "ids")
    fam("person: dismissed reopened bugs", r"dismissed maximum bugs .* now reopened", None,
        lambda p: ids(p.call("bug_board", status="reopened", actor="dismisser", rank="max")), "ids")
    fam("pr: fix PRs of a customer's bugs", r"PR links submitted by (?P<eid>eid_\w+) to resolve bugs reported by (?P<c>\w+)",
        None, lambda p: ids(p.call("fix_trail", customer=p.m["c"], engineer=p.m["eid"])), "ids")
    fam("person: previous release worked on", r"who worked on the previous release", "AuthoredPrevious",
        lambda p: ids(p.call("who_said", subject="previous_release", kind="worked_on", role=role_in(p.text),
                             **{"return": "people"})), "ids")
    fam("person: previous release demos", r"shared demos of the previous version", None,
        lambda p: ids(p.call("who_said", subject="previous_release", kind="demo", **{"return": "people"})), "ids")
    fam("url: previous release demo urls", r"demo URLs shared by team members for the previous version", None,
        lambda p: ids(p.call("who_said", subject="previous_release", kind="demo", **{"return": "urls"})), "ids")
    fam("url: competitor demo urls", r"demo URLs shared by team members for \w+'s competitor", "Observation",
        lambda p: ids(p.call("who_said", subject="competitor", kind="demo", **{"return": "urls"})), "ids")
    fam("person: competitor demos", r"shared demos of \w+'s competitor", "Observation",
        lambda p: ids(p.call("who_said", subject="competitor", kind="demo", **{"return": "people"})), "ids")
    fam("person: competitor insights", r"provided insights on the strengths and weaknesses", "Observation",
        lambda p: sorted(set(ids(p.call("who_said", subject="competitor", kind="feature", **{"return": "people"})))
                         | set(ids(p.call("who_said", subject="competitor", kind="weakness", **{"return": "people"})))),
        "ids")
    fam("person: authors and key reviewers", r"authors and key reviewers of the (?P<d>.+?) for the \w+ product", None,
        lambda p: ids(p.call("doc_people", document_type=p.m["d"], scope="any", **{"return": "people"})), "ids")
    fam("person: approved feature PRs max", r"highest number of approved feature development PRs", "Feature",
        lambda p: ids(p.call("pr_board", kind="feature_development", approval="approved", rank="max",
                             **{"return": "people"})), "ids")
    fam("person: unapproved feature PRs max", r"highest number of unapproved feature development PRs", "Feature",
        lambda p: ids(p.call("pr_board", kind="feature_development", approval="unapproved", rank="max",
                             **{"return": "people"})), "ids")
    fam("person: responsible for unimplemented", r"responsible for features .* not implemented", "Feature",
        lambda p: ids(p.call("feature_fate", fate="unimplemented", **{"return": "people"})), "ids")
    pr_common = lambda p, **kw: ids(p.call("pr_board", feature=feature_in(p.text, p.product), **kw))  # noqa: E731
    fam("pr: not approved", r"PRs .* that were not approved", "Feature",
        lambda p: pr_common(p, approval="unapproved"), "ids")
    fam("pr: reverted, product direction", r"reverted due to changes in product direction", "Feature",
        lambda p: pr_common(p, state="reverted", revert_reason="product_direction"), "ids")
    fam("pr: reverted, introduced bugs", r"reverted (?:because|as) they introduced bugs", "Feature",
        lambda p: pr_common(p, state="reverted", revert_reason="introduced_bugs"), "ids")
    fam("pr: need to be merged", r"need to be merged", "Feature", lambda p: pr_common(p, state="need_merge"), "ids")
    fam("pr: need to be closed", r"need to be closed", "Feature",
        lambda p: pr_common(p, approval="approved", state="need_close"), "ids")
    fam("pr: approved", r"approved PRs for", "Feature", lambda p: pr_common(p, approval="approved"), "ids")
    # content families (key-phrase coverage)
    fam("content: unresolved issues of a company", r"unresolved issues reported by (?P<c>\w+)", None,
        lambda p: p.call("bug_board", status="unresolved", customer=p.m["c"], **{"return": "bugs"}), "text")
    fam("content: no-fix bugs of a company", r"bugs reported by (?P<c>\w+) in \w+ that did not require", None,
        lambda p: p.call("bug_board", status="no_fix_needed", customer=p.m["c"], **{"return": "bugs"}), "text")
    fam("content: suggestions by role", r"changes suggested by (?P<r>.+?) to improve the (?P<d>.+?) for", "Suggested",
        lambda p: p.call("doc_people", document_type=p.m["d"], role=ROLES[p.m["r"].lower()],
                         **{"return": "suggestions"}), "text")
    fam("content: features not added (security)", r"could not be added .* security", "Feature",
        lambda p: p.call("feature_fate", fate="reverted", reason="security"), "text")
    fam("content: features not added (direction)", r"could not be added .* product direction", "Feature",
        lambda p: p.call("feature_fate", fate="reverted", reason="product_direction"), "text")
    fam("content: delayed then added", r"temporarily delayed", "Feature",
        lambda p: p.call("feature_fate", fate="delayed_then_added", reason="product_direction"), "text")
    fam("content: discussed not implemented", r"discussed but not implemented\?$", "Feature",
        lambda p: p.call("feature_fate", fate="unimplemented"), "text")
    fam("content: new features about a topic", r"new features (?:related to|about) (?P<t>.+?) (?:are|have been) added",
        "Feature", lambda p: p.call("feature_fate", fate="implemented", topic=p.m["t"]), "text")
    fam("content: competitor unique features", r"unique features of", "Observation",
        lambda p: p.call("who_said", subject="competitor", kind="feature", **{"return": "claims"}), "text")
    fam("content: competitor weaknesses", r"weaknesses of \w+'s competitor", "Observation",
        lambda p: p.call("who_said", subject="competitor", kind="weakness", **{"return": "claims"}), "text")
    fam("content: previous release customer issues", r"issues were reported by customers for the previous release",
        "Observation", lambda p: p.call("who_said", subject="previous_release", kind="issue", **{"return": "claims"}),
        "text")
    fam("content: previous release QA issues", r"reported during QA testing for the previous release", "Observation",
        lambda p: p.call("who_said", subject="previous_release", kind="qa_issue", **{"return": "claims"}), "text")
    fam("content: previous release strengths", r"strengths were highlighted by customers for the previous release",
        "Observation", lambda p: p.call("who_said", subject="previous_release", kind="strength", **{"return": "claims"}),
        "text")
    return F


def row_text(rows):
    parts = []
    for r in rows:
        parts += [str(r.get("answer", "")), str(r.get("impact_text") or ""), *map(str, r.get("details") or []),
                  str(r.get("section") or "")]
    return "\n".join(p for p in parts if p)


STOP = frozenset("that this with from have were been their there which about would could should into also than "
                 "they them will when what your these those other more some such only over most".split())


def coverage(gold, text):
    items = gold if isinstance(gold, list) else [s for s in re.split(r"[\n;]|(?<=[.!?])\s+", str(gold)) if s.strip()]
    have = set(normalize(text).split())
    got = 0
    for it in items:
        words = {w for w in normalize(str(it)).split() if len(w) > 3 and w not in STOP}
        got += bool(words) and len(words & have) / len(words) >= 0.6
    return 100.0 * got / len(items) if items else 0.0


def present(g, needs):
    """Do the facts a family needs exist in this graph yet?"""
    q = {"Feature": "MATCH (n:Feature) RETURN count(n)", "Observation": "MATCH (n:Observation) RETURN count(n)",
         "Suggested": "MATCH ()-[e:SUGGESTED]->() RETURN count(e)",
         "AuthoredPrevious": "MATCH (d:Document) WHERE d.release_scope = 'previous' RETURN count(d)"}.get(needs)
    return q is None or g.ro_query(q).result_set[0][0] > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="6.0")
    ap.add_argument("--misses", action="store_true", help="list the misses of each family (ids and scores only)")
    args = ap.parse_args(argv)
    graphs = {p: agent_graph(p, args.server) for p in PRODUCTS}
    for g in graphs.values():
        g.server_version = args.server
    fams = families()
    stats = defaultdict(lambda: {"n": 0, "pre": [], "post": [], "calls": [], "misses": [], "pending": 0, "kind": "ids"})
    unmatched = []
    for q in load_questions("dev"):
        if q.key.kind != "ans":
            continue
        text = q.text.replace("’", "'")
        hit = next(((n, rx, needs, fn, kind, rx.search(text)) for n, rx, needs, fn, kind in fams if rx.search(text)), None)
        if hit is None:
            unmatched.append(str(q.key))
            continue
        name, rx, needs, fn, kind, m = hit
        s = stats[name]
        s["kind"], s["type"] = kind, q.type
        s["n"] += 1
        g = graphs[q.key.product]
        if name.endswith("worked on") and role_in(text) == "QA Specialist":
            needs = "Observation"  # named QA reporters come from the semantic layer
        if not present(g, needs):
            s["pending"] += 1
            continue
        p = Plan(g, q.key.product, text)
        p.m = m.groupdict()
        try:
            out = fn(p)
        except ToolError as e:
            out = []
            s["misses"].append(f"{q.key}: tool refused: {e}")
        if kind == "ids":
            proposed = list(dict.fromkeys(out))
            ans_text = ", ".join(proposed)
            g_text, g_ids, dropped = guard(ans_text, proposed, p.results)
            pre, post = 100 * f1_score_sets(q.gold, proposed), 100 * f1_score_sets(q.gold, g_ids)
            if pre < 100 and args.misses:
                extra, lost = sorted(set(proposed) - set(q.gold)), sorted(set(q.gold) - set(proposed))
                s["misses"].append(f"{q.key} F1 {pre:.0f}: predicted {len(proposed)}, gold {len(q.gold)}, "
                                   f"extra {extra[:6]}, missing {lost[:6]}")
        else:
            body = row_text(out)
            pre = coverage(q.gold, body)
            post = coverage(q.gold, guard(body, [], p.results)[0])
            if pre < 100 and args.misses:
                s["misses"].append(f"{q.key} coverage {pre:.0f}: {len(out)} rows")
        s["pre"].append(pre)
        s["post"].append(post)
        s["calls"].append(p.trace.calls)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"server {args.server}; bounded oracle-plan score, mean per family (n scored / n questions)\n")
    print(f"{'family':46} {'n':>7} {'pre-guard':>10} {'post-guard':>10} {'max calls':>9}")
    by_type = defaultdict(lambda: ([], []))
    for name, s in sorted(stats.items()):
        if not s["pre"]:
            print(f"{name:46} {0:>3}/{s['n']:<3} {'pending 4c':>10}")
            continue
        extra = f"  (+{s['pending']} pending 4c)" if s["pending"] else ""
        extra += f"  PARTIAL: {PARTIAL[name]}" if name in PARTIAL else ""
        print(f"{name:46} {len(s['pre']):>3}/{s['n']:<3} {mean(s['pre']):>10.1f} {mean(s['post']):>10.1f} "
              f"{max(s['calls']):>9}{extra}")
        for line in s["misses"]:
            print(f"    miss {line}")
        if name in PARTIAL:
            continue  # not in the type means
        by_type[s["type"] if s["kind"] == "ids" else "content"][0].extend(s["pre"])
        by_type[s["type"] if s["kind"] == "ids" else "content"][1].extend(s["post"])
    print("\nper type, pre/post-guard (scored questions only, partial families left out):", "; ".join(f"{t} {mean(a):.1f}/{mean(b):.1f} (n={len(a)})"
                                                         for t, (a, b) in sorted(by_type.items())))
    if unmatched:
        print("no plan for:", ", ".join(unmatched))
    return 0


if __name__ == "__main__":
    sys.exit(main())
