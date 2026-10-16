"""Layer 2b: bug extraction, cross-session linking and official bug predicates (Task 4b).

Per bug channel, in time order: sessions are parsed by rules (bug_rules); a session goes to the model only when it holds
a lifecycle-looking message the rules could not parse, or a reopening (whose follow-up assignment the rules do not
read). Model items are validated (citations must be verbatim spans of the session's messages, ids must exist) and rules
items win on any message they handled. Bugs register in a per-channel registry by normalised title and alias; events
link by exact or alias match against every prior bug (resolved and dismissed included). The events left over go to one
model call per channel, which may only pick an existing bug that was already reported at that time, or null. A null
link is kept and counted, never turned into a new bug. Official predicates are derived afterwards (bugstate).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime

from inherit import llm
from inherit.llm import BudgetExceeded
from inherit.extract import bug_rules as br
from inherit.extract.bugstate import derive_bug, sort_events
from inherit.extract.catalog import build_catalog, repo_scope
from inherit.extract.cite import SourceRecord, normalise, validate_citation, validate_items
from inherit.extract.llm_extract import Caller
from inherit.extract.prompts import BUG_LINK_SYSTEM, BUG_SESSION_SYSTEM, BUGS_PROMPT_VERSION
from inherit.extract.sessions import sessions
from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge

EXTRACTOR = "bugs/v2"
PROMPT_VERSION = BUGS_PROMPT_VERSION
ALLOWED = {("message", "text")}
EVENT_TYPES = ("REPORTED", "ASSIGNED", "RESOLVED", "DISMISSED", "REOPENED")
MODEL_EVENT_TYPES = EVENT_TYPES[1:]
TYPE_RANK = {t: i for i, t in enumerate(EVENT_TYPES)}
EID_RE = re.compile(r"\beid_[A-Za-z0-9_]+")
HELP_REQUEST = re.compile(r"\b(?:assist|help|support) (?:me|us)\b|\btwo heads\b|\bpair (?:up|with)\b", re.I)
# Wording an actor message must contain for a model event of that type (the model over-reads follow-up chatter).
EVIDENCE = {
    "RESOLVED": re.compile(r"resolv|\bfix|patch|deploy|\bmerged?\b|releas|shipp|\bclosed?\b|pull/\d+", re.I),
    "DISMISSED": re.compile(br.NOFIX.pattern + r"|no action|dismiss|\bclos|not (?:going|planning) to", re.I),
    "REOPENED": re.compile(r"re-?open|\bagain\b|\bstill\b|resurfac|persist|recur|regress|came back|not (?:fixed|resolved)",
                           re.I),
}


@dataclass
class BugResult:
    facts: Facts
    report: dict


def _is_fatal(exc: Exception) -> bool:
    """Budget exhaustion, an unreachable proxy and client timeouts stop the layer; they are not per-unit failures."""
    return isinstance(exc, (BudgetExceeded, TimeoutError)) or (
        isinstance(exc, RuntimeError) and str(exc).startswith("metering proxy"))


def _dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


# ---------------------------------------------------------------- prompts

def _msg_line(m, employees) -> str:
    name = (employees.get(m.user_id) or {}).get("name") or "unknown"
    return f"[{m.id}] {m.ts} {name} ({m.user_id}): {m.text}"


def build_session_request(session, parse, rec, cat, known_areas=()) -> tuple[list[dict], int]:
    """(chat messages, max_tokens) for one session. The model sees the numbered messages, what the rules already
    parsed, the messages they could not parse and the catalogue entries for PR links found in the text."""
    msgs = [m for m in session if not m.is_bot]
    lines = [f"PRODUCT: {rec.product}" + (f" (also called: {', '.join(cat.aliases)})" if cat.aliases else ""),
             f"CHANNEL: {msgs[0].channel_name} (the name after 'bug-' is the channel owner, not an assignee)",
             "", "MESSAGES (time order):"]
    lines += [_msg_line(m, rec.employees) for m in msgs]
    lines += ["", "ALREADY PARSED BY RULES (do not repeat):"]
    hint_refs = hints_of(parse)
    show = parse.show_ids
    for b in parse.bugs:
        if show is not None and b["message_id"] not in show:
            continue
        lines.append(f"- BUG {b['local_id']} '{b['title']}' introduced in {b['message_id']}; "
                     f"customers {', '.join(b['customer_ids']) or 'none'}")
    for e in parse.events:
        if show is not None and e["message_id"] not in show:
            continue
        ref = next((r for r, ev in hint_refs.items() if ev is e), e.get("local_bug"))
        what = f"ref {ref}" + (f" = '{e['mention']}'" if e.get("mention") and ref in hint_refs else "")
        lines.append(f"- EVENT {e['type']} in {e['message_id']} about {what}"
                     + (f"; assignees {', '.join(e['assignee_eids'])}" if e["assignee_eids"] else ""))
    for x in parse.externals:
        if show is not None and x["message_id"] not in show:
            continue
        lines.append(f"- OPEN-SOURCE look-alike '{x['title']}' in {x['message_id']}")
    if not (parse.bugs or parse.events or parse.externals):
        lines.append("- nothing")
    lines += ["", "MESSAGES THE RULES COULD NOT PARSE (look here first): " + (", ".join(parse.residual) or "none")]
    if parse.no_area and not (parse.residual or parse.reopen):
        lines += ["", "Only the product areas are needed for this session: return empty bugs, events and external "
                      "arrays."]
    if parse.reopen:
        lines.append("This session contains a reopening. Report who is assigned to or takes the bug afterwards, and "
                     "any resolution or dismissal stated in the follow-up messages.")
    if parse.no_area:
        by_lid = {b["local_id"]: b for b in parse.bugs}
        lines += ["", "BUGS WITHOUT A PRODUCT AREA (report `components` for these):"]
        lines += [f"- {lid} '{by_lid[lid]['title']}' in {by_lid[lid]['message_id']}" for lid in parse.no_area]
        lines.append("KNOWN AREAS in this product: " + (", ".join(known_areas) or "none yet"))
    text = "\n".join(m.text for m in msgs)
    prs = [p for p in cat.prs if p.link in set(br.pr_links(text))]
    if prs:
        lines += ["", "PRODUCT PULL REQUESTS LINKED IN THESE MESSAGES:"] + [f"- {p.link} {p.title}" for p in prs]
    n = len(parse.residual) + (2 if parse.reopen else 0) + len(parse.no_area)
    max_tokens = min(4000, 600 + 350 * n)
    return [{"role": "system", "content": BUG_SESSION_SYSTEM}, {"role": "user", "content": "\n".join(lines)}], max_tokens


MAX_INPUT_TOKENS = 6000


def split_session(session, parse, rec, cat, known_areas=()):
    """[(messages, parse)] to send. A session whose request fits in 6,000 input tokens (characters / 3) goes whole.
    A longer one is cut at message boundaries so that every assembled chunk request (system prompt, opener, rules
    lines, messages) fits. Each chunk starts with the session opener as context; refs m1.. stay stable because they
    come from the full rules list, but only the rules lines that belong to the chunk's messages are shown."""
    def size(req):
        return (len(req[0]["content"]) + len(req[1]["content"])) / 3

    if size(build_session_request(session, parse, rec, cat, known_areas)[0]) <= MAX_INPUT_TOKENS:
        return [(session, parse)]
    msgs = [m for m in session if not m.is_bot]
    by_msg = {b["local_id"]: b["message_id"] for b in parse.bugs}

    def chunk_parse(c):
        ids = {m.id for m in c}
        return replace(parse, residual=[x for x in parse.residual if x in ids],
                       no_area=[x for x in parse.no_area if by_msg[x] in ids], show_ids=ids)

    def with_opener(c):
        return c if c[0].id == msgs[0].id else [msgs[0], *c]

    chunks, cur = [], []
    for m in msgs:
        trial = cur + [m]
        if cur and size(build_session_request(with_opener(trial), chunk_parse(trial), rec, cat, known_areas)[0]) \
                > MAX_INPUT_TOKENS:
            chunks.append(cur)
            trial = [m]
        cur = trial
    chunks.append(cur)
    return [(with_opener(c), chunk_parse(c)) for c in chunks]


def _is_str(v):
    return isinstance(v, str)


def _str_list(v):
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _citations(v):
    return isinstance(v, list) and all(
        isinstance(c, dict) and all(c.get(k) is None or isinstance(c.get(k), str) for k in ("record_id", "quote", "field"))
        and (c.get("position") is None or isinstance(c.get("position"), int)) for c in v)


# field -> type check, applied only when the field is present and not null (a missing field is handled by the cleaner)
_SHAPES = {
    "bugs": {"local_id": _is_str, "title": _is_str, "message_id": _is_str, "aliases": _str_list,
             "component": _is_str, "impact_text": _is_str, "customer_ids": _str_list, "citations": _citations},
    "events": {"type": _is_str, "bug_ref": _is_str, "bug_mention": _is_str, "actor_message_id": _is_str,
               "assignee_eids": _str_list, "fix_pr_links": _str_list, "citations": _citations},
    "external": {"title": _is_str, "link": _is_str, "customer_ids": _str_list, "similar_mention": _is_str, "message_id": _is_str,
                 "citations": _citations},
    "components": {"bug_ref": _is_str, "component": _is_str, "citations": _citations},
    "links": {"item": _is_str, "bug": _is_str},
}


def shape_error(data: dict, keys) -> str | None:
    """First type violation in the fields the code consumes, else None. Arrays must be lists of objects."""
    for key in keys:
        rows = data.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):
            return f"{key} is not a list of objects"
        for i, row in enumerate(rows):
            for field, ok in _SHAPES[key].items():
                v = row.get(field)
                if v is not None and not ok(v):
                    return f"{key}[{i}].{field} has the wrong type"
    return None


def session_shape_error(data: dict) -> str | None:
    return shape_error(data, ("bugs", "events", "external", "components"))


def link_shape_error(data: dict) -> str | None:
    return shape_error(data, ("links",))


def hints_of(parse) -> dict[str, dict]:
    """Refs m1.. for rules events that name a bug by words (no local bug): the model may point at them."""
    out = {}
    for e in parse.events:
        if e.get("local_bug") is None and e.get("mention"):
            out[f"m{len(out) + 1}"] = e
    return out


def build_link_request(bugs_in_order, items, channel_name) -> tuple[list[dict], int]:
    lines = [f"CHANNEL: {channel_name}", "", "BUGS:"]
    for ref, b in bugs_in_order:
        lines.append(f"{ref}: '{b['title']}'; aliases {b['aliases'][:5]}; component {b['component']}; "
                     f"customers {', '.join(b['customer_ids']) or 'none'}; first reported {b['first_ts']}; "
                     f"text: {(b.get('impact_text') or '')[:220]!r}")
    lines += ["", "ITEMS:"]
    for ref, it in items:
        lines.append(f"{ref}: kind {it['kind']}; type {it['type']}; time {it['ts']}; words used for the bug: "
                     f"'{it['mention']}'")
        for t in it["context"]:
            lines.append(f"    message: {t}")
    return ([{"role": "system", "content": BUG_LINK_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
            min(4000, 200 + 60 * len(items)))


# ---------------------------------------------------------------- model answer -> validated candidates

def _clean_session_answer(data, session, parse, rec, cat, rep, sess_src, areas_only=False, prefix=""):
    """Validated (bugs, events, externals) from one model answer. Counts every drop in `rep`."""
    msgs = {m.id: m for m in session if not m.is_bot}
    people = {m.user_id for m in msgs.values()} | {x for m in msgs.values() for x in m.mentions}
    text = "\n".join(m.text for m in msgs.values())
    session_links = set(br.pr_links(text))
    look_alike_session = any(repo_scope(x, cat.product_repos) == "external" for x in session_links)
    dropped = rep["dropped"]

    def drop(kind, reason):
        dropped[kind][reason] += 1

    if areas_only:  # the rules parsed everything else: whatever the model adds here is ignored, never merged
        rep["ignored_model_events"] += sum(1 for e in (data.get("events") or []) if isinstance(e, dict))
        data = {"components": data.get("components")}
    residual = set(parse.residual)
    bugs_raw = [_repaired(b, msgs, sess_src, rep, residual) for b in (data.get("bugs") or []) if isinstance(b, dict)]
    ok_bugs, bad = validate_items(bugs_raw, sess_src, ALLOWED)
    for reason, n in bad.items():
        dropped["bugs"][reason] += n
    bugs = []
    rejected: dict[str, str] = {}  # local id -> title of a lifecycle message's invented bug: its events keep the words
    for b in ok_bugs:
        title, mid = str(b.get("title") or "").strip(), b.get("message_id")
        if not title or mid not in msgs:
            drop("bugs", "bad_title_or_message")
            continue
        if br.LIFECYCLE.search(msgs[mid].text):
            drop("bugs", "lifecycle_message")  # a resolution, dismissal or reopening names a bug, it does not open one
            if isinstance(b.get("local_id"), str):
                rejected[b["local_id"]] = title
            continue
        if mid in parse_handled(parse):
            drop("bugs", "already_parsed")
            continue
        if any(repo_scope(x, cat.product_repos) == "external" for x in br.pr_links(msgs[mid].text)):
            drop("bugs", "look_alike_message")  # an open-source look-alike message never introduces a customer bug
            continue
        cids = [c for c in dict.fromkeys(b.get("customer_ids") or []) if isinstance(c, str) and c in text]
        comp = b.get("component") if isinstance(b.get("component"), str) and b.get("component").strip() else None
        if comp and not area_in(comp, msgs[mid].text):
            drop("components", "not_in_message")
            comp = None
        bugs.append({"local_id": f"x:{prefix}{b.get('local_id')}", "title": title,
                     "aliases": [a for a in (b.get("aliases") or []) if isinstance(a, str) and a.strip()],
                     "component": comp,
                     "impact_text": str(b.get("impact_text") or ""), "customer_ids": cids, "message_id": mid,
                     "citations": b["citations"], "source": "model"})

    ev_raw = [e for e in (data.get("events") or []) if isinstance(e, dict)]
    ok_ev, bad = validate_items(ev_raw, sess_src, ALLOWED)
    for reason, n in bad.items():
        dropped["events"][reason] += n
    events = []
    handled = parse_handled(parse)
    hints = hints_of(parse)
    local_ids = {b["local_id"] for b in parse.bugs} | {b["local_id"] for b in bugs}
    for e in ok_ev:
        typ, mid = e.get("type"), e.get("actor_message_id")
        if typ not in MODEL_EVENT_TYPES or mid not in msgs:
            drop("events", "bad_type_or_message")
            continue
        if mid in handled:
            rep["ignored_model_events"] += 1
            continue
        if typ == "ASSIGNED" and look_alike_session:  # people reviewing open-source PRs are not assigned to the bug
            drop("events", "look_alike_session")
            continue
        if typ == "ASSIGNED" and HELP_REQUEST.search(msgs[mid].text):
            drop("events", "help_request")  # asking a colleague to assist does not hand over the bug
            continue
        if typ == "ASSIGNED" and _answers_help_request(msgs, mid):
            drop("events", "help_reply")  # the colleague who agrees to assist is a helper, not the assignee
            continue
        if typ in EVIDENCE and not EVIDENCE[typ].search(msgs[mid].text):
            drop("events", "no_evidence_for_type")
            continue
        assignees = [a for a in dict.fromkeys(e.get("assignee_eids") or []) if isinstance(a, str)]
        if typ == "ASSIGNED" and not assignees:
            drop("events", "no_assignee")
            continue
        if any(a not in people for a in assignees):
            drop("events", "assignee_not_in_session")
            continue
        fix = [x for x in dict.fromkeys(e.get("fix_pr_links") or []) if isinstance(x, str)]
        fix = [br.pr_links(x)[0] for x in fix if br.pr_links(x)]
        fix = [x for x in fix if x in session_links]
        ref, mention, local = e.get("bug_ref"), e.get("bug_mention"), None
        if isinstance(ref, str) and f"x:{prefix}{ref}" in local_ids:
            local = f"x:{prefix}{ref}"
        elif isinstance(ref, str) and ref in local_ids:
            local = ref
        elif isinstance(ref, str) and ref in hints:
            mention = hints[ref]["mention"]
        elif isinstance(ref, str) and ref in rejected:
            mention = rejected[ref]
        if local is None and not (isinstance(mention, str) and mention.strip()) and isinstance(ref, str) \
                and len(ref.split()) >= 2:
            mention = ref  # the model wrote the bug's words where a ref belongs
        if local is None and not (isinstance(mention, str) and mention.strip()):
            drop("events", "no_bug_reference")
            continue
        if local is None and normalise(mention) not in normalise(text) and ref not in hints and ref not in rejected:
            drop("events", "mention_not_in_session")
            continue
        events.append({"type": typ, "message_id": mid, "mention": mention if local is None else None,
                       "local_bug": local, "assignee_eids": assignees, "fix_pr_links": fix,
                       "citations": e["citations"], "source": "model"})

    ex_raw = [_repaired(x, msgs, sess_src, rep, set()) for x in (data.get("external") or []) if isinstance(x, dict)]
    ok_ex, bad = validate_items(ex_raw, sess_src, ALLOWED)
    for reason, n in bad.items():
        dropped["external"][reason] += n
    externals = []
    for x in ok_ex:
        link = br.pr_links(str(x.get("link") or ""))
        mid = x.get("message_id")
        if not link or link[0] not in session_links or mid not in msgs or mid in handled \
                or not str(x.get("title") or "").strip():
            drop("external", "bad_link_or_message")
            continue
        sim = x.get("similar_mention")
        externals.append({"link": link[0], "title": str(x["title"]).strip(), "project": "/".join(br.repo_of(link[0])),
                          "similar_mention": sim if isinstance(sim, str) and sim.strip() else None,
                          "message_id": mid, "citations": x["citations"], "source": "model"})
    areas, late = {}, {}
    by_lid = {b["local_id"]: b for b in parse.bugs}
    comp_raw = [_repointed(c, msgs, parse, by_lid, sess_src, rep) for c in (data.get("components") or [])
                if isinstance(c, dict)]
    ok_comp, bad = validate_items(comp_raw, sess_src, ALLOWED)
    for reason, n in bad.items():
        dropped["components"][reason] += n
    for c in ok_comp:
        lid, comp = c.get("bug_ref"), c.get("component")
        if lid not in parse.no_area or not isinstance(comp, str) or not comp.strip():
            drop("components", "bad_ref_or_empty")
        elif area_in(comp, "\n".join(m.text for m in earlier_or_same(msgs, by_lid[lid]["message_id"]))):
            areas[lid] = comp.strip()
        elif area_in(comp, "\n".join(m.text for m in acceptance_of(msgs, parse, by_lid[lid]["message_id"]))):
            late[lid] = comp.strip()  # named only in the acceptance: applied after the link calls (see run_bugs)
        else:
            drop("components", "not_in_message")
    return bugs, events, externals, areas, late


def canonical_areas(variants) -> dict[str, str]:
    """area key -> display form. Spellings that differ only by case or `&` versus `and` fold into one: the capitalised
    form, then the most frequent, then the one with `&`."""
    count: Counter = Counter(v for v in variants if v)
    best: dict[str, str] = {}
    for v in sorted(count, key=lambda v: (not v[:1].isupper(), -count[v], "&" not in v, v)):
        best.setdefault(br.area_key(v), v)
    return best


def _answers_help_request(msgs: dict, message_id: str) -> bool:
    """True when the message before `message_id` asks its author (by @mention) to help or assist."""
    prev = None
    for m in msgs.values():
        if m.id == message_id:
            return bool(prev and HELP_REQUEST.search(prev.text) and msgs[message_id].user_id in prev.mentions)
        prev = m
    return False


def _assignment_sentence(text: str, eid: str) -> str | None:
    """The sentence of `text` that carries an assignment cue and @mentions `eid` (the evidence of a hand-off)."""
    start, spans = 0, []
    for b in br.SENT_BOUNDARY.finditer(text):
        spans.append((start, b.start()))
        start = b.end()
    spans.append((start, len(text)))
    for a, b in spans:
        sent = text[a:b]
        if br.ASSIGN_CUE.search(sent) and f"@{eid}" in sent:
            return sent.strip()
    return None


def place_assignments(events, msgs, pos, src_all, counts) -> None:
    """One ASSIGNED event per assignment, on the message that hands the bug over. The model puts the event on the
    message where the assignee accepts. When that person accepts before the next bug is introduced and an earlier
    message of the same session hands them the bug (@mentions them in a sentence with an assignment cue), the event
    moves to that hand-off message and keeps its type and assignee; if the hand-off already has its own ASSIGNED event
    for that person, the acceptance duplicate is dropped. A bare mention is never a hand-off, a hand-off in another
    session is never used (a later reassignment is not back-filled onto the first), and when nobody hands the bug over
    the event stays on the message where the person takes it. Runs after the channel link call so no request changes."""
    sess_of: dict[str, int] = {}
    first_pos: dict[int, int] = {}
    for k, sess in enumerate(sessions(msgs)):
        for m in sess:
            sess_of[m.id] = k
            first_pos[k] = min(first_pos.get(k, pos[m.id]), pos[m.id])
    intro = {e["bug_id"]: pos[e["message_id"]] for e in events if e["type"] == "REPORTED" and e.get("bug_id")}
    intro_pos = sorted(set(intro.values()))
    gone: set[int] = set()
    for e in sorted((x for x in events if x["type"] == "ASSIGNED" and x.get("bug_id")),
                    key=lambda x: (pos[x["message_id"]], TYPE_RANK[x["type"]])):
        who = e["assignee_eids"]
        am = msgs[pos[e["message_id"]]]
        if len(who) != 1 or am.is_bot or am.user_id != who[0]:
            continue  # only a person's own acceptance can be a duplicate of, or sit after, a hand-off
        x, ai, k = who[0], pos[e["message_id"]], sess_of[e["message_id"]]
        ri = intro.get(e["bug_id"])
        if ri is not None and sess_of[msgs[ri].id] == k and ri <= ai:
            if any(ri < p <= ai for p in intro_pos):
                continue  # the acceptance answers a later bug introduction
            lo = ri
        else:  # the bug was introduced in an earlier session: this is a reassignment inside a session of its own
            if any(first_pos[k] <= p <= ai for p in intro_pos):
                continue
            lo = first_pos[k]
        same = [o for o in events if o is not e and id(o) not in gone and o["type"] == "ASSIGNED"
                and o.get("bug_id") == e["bug_id"] and x in o["assignee_eids"] and lo <= pos[o["message_id"]] < ai]
        if same:
            gone.add(id(e))
            counts["duplicate_dropped"] += 1
            continue
        hand = next((msgs[p] for p in range(ai - 1, lo - 1, -1)
                     if not msgs[p].is_bot and msgs[p].user_id != x and x in br.assignees_in(msgs[p].text, msgs[p].user_id)),
                    None)
        if hand is None:
            continue
        if any(o is not e and o.get("bug_id") == e["bug_id"] and o["type"] not in ("REPORTED", "ASSIGNED")
               and pos[hand.id] <= pos[o["message_id"]] < ai for o in events):
            continue  # the bug changed state in between: not a plain hand-off
        sent = _assignment_sentence(hand.text, x)
        good, _ = validate_items([{"citations": [{"record_id": hand.id, "quote": sent}]}], [src_all[hand.id]], ALLOWED) \
            if sent else ([], {})
        if not good:
            continue
        e["message_id"], e["ts"], e["citations"] = hand.id, hand.ts, good[0]["citations"]
        counts["moved_to_handoff"] += 1
    events[:] = [e for e in events if id(e) not in gone]


def _repointed(c: dict, msgs: dict, parse, by_lid: dict, sess_src, rep) -> dict:
    """A component whose quote is verbatim in the bug's uniquely linked acceptance message but is cited on another
    record (the model names the intro message for words the answer wrote) is cited on the acceptance message."""
    cites, bug = c.get("citations"), by_lid.get(c.get("bug_ref"))
    if bug is None or not isinstance(cites, list) or not all(isinstance(x, dict) for x in cites):
        return c
    if any(validate_citation(x, sess_src, ALLOWED).ok for x in cites):
        return c
    acc = acceptance_of(msgs, parse, bug["message_id"])
    if not acc:
        return c
    fixed = [{**x, "record_id": acc[0].id} if normalise(x.get("quote")) and normalise(x["quote"]) in normalise(acc[0].text)
             else x for x in cites]
    if fixed != cites:
        rep["repaired_citations"] += 1
    return {**c, "citations": fixed}


def acceptance_of(msgs: dict, parse, message_id: str) -> list:
    """The one message that answers the bug introduced in `message_id`: after it, before the next bug introduction of
    the session, written by someone other than the introducer. [] when there is none or more than one (then it is not
    uniquely linked and its wording is not evidence for this bug's area)."""
    order = list(msgs)
    if message_id not in order:
        return []
    start = order.index(message_id)
    intros = sorted(order.index(b["message_id"]) for b in parse.bugs if b["message_id"] in order)
    end = next((i for i in intros if i > start), len(order))
    author = msgs[message_id].user_id
    window = [msgs[k] for k in order[start + 1:end] if msgs[k].user_id != author]
    return window if len(window) == 1 else []


def earlier_or_same(msgs: dict, message_id: str) -> list:
    """The session's messages up to and including `message_id` (an area is often named in an earlier message)."""
    out = []
    for m in msgs.values():
        out.append(m)
        if m.id == message_id:
            break
    return out


def area_in(component: str, text: str) -> bool:
    k = br.area_key(component)
    return bool(k) and f" {k} " in f" {br.area_key(text)} "


def _repaired(item: dict, msgs, sess_src, rep, residual) -> dict:
    """A bug or look-alike whose citations all fail but whose title is a verbatim span of its own message (the model
    usually joined sentences or dropped markup in the quote) is cited by the title itself. A bug whose title the model
    reworded is cited by its customer id when the message is an intake the rules left unparsed. Both are counted."""
    mid, title = item.get("message_id"), str(item.get("title") or "").strip()
    if mid not in msgs or not title:
        return item
    link = br.pr_links(str(item.get("link") or ""))
    if any(validate_citation(c, sess_src, ALLOWED).ok for c in item.get("citations") or [] if isinstance(c, dict)):
        return item
    if normalise(title) in normalise(msgs[mid].text):
        rep["repaired_citations"] += 1
        return {**item, "citations": [{"record_id": mid, "quote": title}]}
    if link and link[0] in msgs[mid].text and "customer_ids" not in item:  # a look-alike: its link is the evidence
        rep["repaired_citations"] += 1
        return {**item, "citations": [{"record_id": mid, "quote": link[0]}]}
    cust = [c for c in item.get("customer_ids") or [] if isinstance(c, str) and c in msgs[mid].text]
    if mid in residual and cust:  # a message the rules flagged as an unparsed intake, naming that customer
        rep["repaired_citations"] += 1
        return {**item, "citations": [{"record_id": mid, "quote": cust[0]}]}
    return item


def parse_handled(parse) -> set:
    return parse.handled


# ---------------------------------------------------------------- the run

def run_bugs(rec, *, chat=llm.chat, cache_root=None, dry: bool = False) -> BugResult:
    cat = build_catalog(rec)
    names = [rec.product, *cat.aliases]
    caller = Caller(chat, cache_root, PROMPT_VERSION, dry=dry)
    bug_msgs = [m for m in rec.messages if m.bug_channel_owner]
    by_channel: dict[str, list] = defaultdict(list)
    for m in bug_msgs:
        by_channel[m.channel_id].append(m)
    by_id = {m.id: m for m in bug_msgs}
    rep = {"dropped": defaultdict(Counter), "ignored_model_events": 0, "extraction_failures": [],
           "unmatched_events": [], "fix_links_not_in_catalogue": [], "sessions": 0, "model_sessions": 0,
           "residual_messages": 0, "unlinked_external": [], "rules_dropped": Counter(), "repaired_citations": 0, "model_calls_split_sessions": 0,
           "placement": Counter()}
    src_all = {m.id: SourceRecord(m.id, "message", "text", m.text) for m in bug_msgs}

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    people: set[str] = set()
    customers_seen: set[str] = set()

    def edge(type_, src, dst, ids, key="", **props):
        e = Edge(type_, src, dst, props, tuple(ids), EXTRACTOR, key)
        check_edge(e)
        edges.append(e)

    all_bugs: dict[str, dict] = {}
    all_events: list[dict] = []
    all_externals: list[dict] = []

    known_areas = list(canonical_areas(
        b["component"] for msgs in by_channel.values()
        for sess in sessions(sorted(msgs, key=lambda m: _dt(m.ts)))
        for b in br.parse_session(sess, cat, rec.customers).bugs if b["component"]).values())

    for channel_id, msgs in by_channel.items():
        msgs = sorted(msgs, key=lambda m: _dt(m.ts))
        pos = {m.id: i for i, m in enumerate(msgs)}
        cname = msgs[0].channel_name
        registry: dict[str, str] = {}
        ch_bugs: dict[str, dict] = {}
        ch_events: list[dict] = []
        ch_externals: list[dict] = []
        pending: list[dict] = []
        late_components: list[tuple[str, str]] = []

        def key_of(text):
            return br.norm_title(text, names)

        def register(bug_id, text):
            k = key_of(text)
            if k and registry.setdefault(k, bug_id) != bug_id:
                return False
            return True

        def eligible(bid, ts):
            return _dt(ch_bugs[bid]["first_ts"]) <= _dt(ts)

        def find_bug(text, ts):
            """(bug id, method) or (None, None): the exact normalised title or an established alias of a bug that was
            already reported at `ts`. Word overlap never decides identity; the rest goes to the channel linker."""
            k = key_of(text)
            bid = registry.get(k)
            if bid and eligible(bid, ts):
                return bid, ("exact" if key_of(ch_bugs[bid]["title"]) == k else "alias")
            return None, None

        for si, session in enumerate(sessions(msgs)):
            try:
                rep["sessions"] += 1
                parse = br.parse_session(session, cat, rec.customers)
                sess_src = [src_all[m.id] for m in session if not m.is_bot]
                # rules items pass the same validator as model items (spans for every fact)
                for kind, attr in (("bugs", "bugs"), ("events", "events"), ("external", "externals")):
                    kept, bad = validate_items(getattr(parse, attr), sess_src, ALLOWED)
                    for r, n in bad.items():
                        rep["rules_dropped"][f"{kind}:{r}"] += n
                    setattr(parse, attr, kept)
                rep["residual_messages"] += len(parse.residual)
                m_bugs, m_events, m_ext = [], [], []
                late_area: dict[str, str] = {}
                if parse.residual or parse.reopen or parse.no_area:
                    rep["model_sessions"] += 1
                    chunks = split_session(session, parse, rec, cat, known_areas)
                    rep["model_calls_split_sessions"] += len(chunks) > 1
                    for ci, (csess, cparse) in enumerate(chunks):
                        req, mt = build_session_request(csess, cparse, rec, cat, known_areas)
                        data, reason = caller.json_call("session", req, mt, session_shape_error)
                        if data is None:
                            rep["extraction_failures"].append({"kind": "session", "channel_id": channel_id,
                                                               "messages": [m.id for m in csess if not m.is_bot],
                                                               "reason": reason})
                            continue
                        csrc = [src_all[m.id] for m in csess if not m.is_bot]
                        cb, ce, cx, areas, late = _clean_session_answer(
                            data, csess, cparse, rec, cat, rep, csrc,
                            areas_only=not (cparse.residual or cparse.reopen),
                            prefix=f"{ci}." if len(chunks) > 1 else "")
                        m_bugs += cb
                        m_events += ce
                        m_ext += cx
                        for b in parse.bugs:
                            b["component"] = b["component"] or areas.get(b["local_id"])
                        late_area.update(late)
                local_bug: dict[str, str] = {}
                # ---- bugs: register, merge with prior bugs by title or alias
                for b in [*parse.bugs, *m_bugs]:
                    m = by_id[b["message_id"]]
                    bug_id = find_bug(b["title"], m.ts)[0] or f"Bug:{rec.product}:{channel_id}:{br.slug(b['title'])}"
                    local_bug[b["local_id"]] = bug_id
                    if not b["component"] and late_area.get(b["local_id"]):
                        late_components.append((bug_id, late_area[b["local_id"]]))
                    if bug_id in ch_bugs:
                        tgt = ch_bugs[bug_id]
                        tgt["customer_ids"] += [c for c in b["customer_ids"] if c not in tgt["customer_ids"]]
                        tgt["component"] = tgt["component"] or b["component"]
                        for a in [b["title"], *b["aliases"]]:
                            if a != tgt["title"] and a not in tgt["aliases"] and register(bug_id, a):
                                tgt["aliases"].append(a)
                        continue
                    ch_bugs[bug_id] = {"id": bug_id, "title": b["title"], "aliases": [], "component": b["component"],
                                       "impact_text": b["impact_text"], "customer_ids": list(b["customer_ids"]),
                                       "first_ts": m.ts, "message_id": m.id, "channel_id": channel_id,
                                       "channel_name": cname}
                    register(bug_id, b["title"])
                    for a in b["aliases"]:
                        if a != b["title"] and register(bug_id, a):
                            ch_bugs[bug_id]["aliases"].append(a)
                    ch_events.append({"type": "REPORTED", "message_id": m.id, "bug_id": bug_id, "assignee_eids": [],
                                      "fix_pr_links": [], "citations": b["citations"], "source": b["source"],
                                      "link_method": "intro", "ts": m.ts})
                # ---- events and look-alikes: link by local bug, then exact or alias, else pending for the channel call
                for e in [*parse.events, *m_events]:
                    m = by_id[e["message_id"]]
                    ev = {**e, "ts": m.ts, "bug_id": None, "link_method": None}
                    if e.get("local_bug"):
                        ev["bug_id"] = local_bug.get(e["local_bug"])
                        ev["link_method"] = "local"
                        if ev["bug_id"] is None:
                            rep["dropped"]["events"]["bug_dropped"] += 1
                            continue
                        if not eligible(ev["bug_id"], m.ts):  # the bug was not reported yet: no target, stays unmatched
                            ev["bug_id"] = ev["link_method"] = None
                    else:
                        ev["bug_id"], ev["link_method"] = find_bug(e["mention"], m.ts)
                    if ev["bug_id"] is None and ev.get("mention"):
                        pending.append(ev)
                    ch_events.append(ev)
                for x in [*parse.externals, *m_ext]:
                    m = by_id[x["message_id"]]
                    xe = {**x, "ts": m.ts, "bug_id": None}
                    if x["similar_mention"]:
                        xe["bug_id"] = find_bug(x["similar_mention"], m.ts)[0]
                    if xe["bug_id"] is None:
                        pending.append({"_external": xe, "type": "SIMILAR", "mention": x["similar_mention"]
                                        or by_id[x["message_id"]].text, "ts": m.ts, "message_id": m.id})
                    ch_externals.append(xe)
            except Exception as exc:  # one bad session never aborts the product
                if _is_fatal(exc):
                    raise
                rep["extraction_failures"].append({"kind": "session", "channel_id": channel_id,
                                                   "messages": [m.id for m in session if not m.is_bot],
                                                   "reason": "exception", "error": f"{type(exc).__name__}: {exc}"})

        # ---- one model call per channel for what is still unmatched
        if pending:
            try:
                ordered_bugs = sorted(ch_bugs.values(), key=lambda b: (_dt(b["first_ts"]), b["id"]))
                bref = {f"B{i + 1}": b for i, b in enumerate(ordered_bugs)}
                items = []
                for p in sorted(pending, key=lambda p: (_dt(p["ts"]), p["message_id"])):
                    eligible = [r for r, b in bref.items() if _dt(b["first_ts"]) <= _dt(p["ts"])]
                    if not eligible:
                        continue
                    cites = [c["record_id"] for c in (p.get("citations") or p["_external"]["citations"])]
                    ctx = [by_id[c].text[:500] for c in dict.fromkeys(cites) if c in by_id][:3]
                    items.append((f"I{len(items) + 1}", {"kind": "external" if "_external" in p else "event",
                                                         "type": p["type"], "mention": p["mention"], "ts": p["ts"],
                                                         "context": ctx, "_p": p, "_ok": set(eligible)}))
                if items:
                    req, mt = build_link_request(list(bref.items()), [(r, i) for r, i in items], cname)
                    data, reason = caller.json_call("link", req, mt, link_shape_error)
                    if data is None:
                        rep["extraction_failures"].append({"kind": "link", "channel_id": channel_id,
                                                           "messages": [i["_p"]["message_id"] for _, i in items],
                                                           "reason": reason})
                    else:
                        answers = {}
                        for row in data.get("links") or []:
                            if isinstance(row, dict) and isinstance(row.get("item"), str):
                                answers[row["item"]] = row.get("bug")
                        for ref, it in items:
                            target = answers.get(ref)
                            if target not in it["_ok"]:
                                continue
                            bug = bref[target]
                            p = it["_p"]
                            if "_external" in p:
                                p["_external"]["bug_id"] = bug["id"]
                            else:
                                p["bug_id"] = bug["id"]
                                p["link_method"] = "model"
                                if p["mention"] != bug["title"] and p["mention"] not in bug["aliases"]:
                                    bug["aliases"].append(p["mention"])
            except Exception as exc:  # a failed link call leaves its items unmatched
                if _is_fatal(exc):
                    raise
                rep["extraction_failures"].append({"kind": "link", "channel_id": channel_id,
                                                   "messages": [q.get("message_id") for q in pending],
                                                   "reason": "exception", "error": f"{type(exc).__name__}: {exc}"})
        # applied after the link call: the link request lists each bug's component, and these must not change it
        for bug_id, comp in late_components:
            ch_bugs[bug_id]["component"] = ch_bugs[bug_id]["component"] or comp
        place_assignments(ch_events, msgs, pos, src_all, rep["placement"])
        all_bugs.update(ch_bugs)
        all_events += [dict(e, msg_pos=pos[e["message_id"]], channel_id=channel_id) for e in ch_events]
        all_externals += ch_externals

    canon = canonical_areas(b["component"] for b in all_bugs.values())
    for b in all_bugs.values():
        if b["component"]:
            b["component"] = canon[br.area_key(b["component"])]

    # ---------------------------------------------------------------- numbering, facts, derived states
    by_msg: dict[str, list[dict]] = defaultdict(list)
    for e in all_events:
        by_msg[e["message_id"]].append(e)
    for mid, es in by_msg.items():
        for n, e in enumerate(sorted(es, key=lambda e: TYPE_RANK[e["type"]])):
            e["n"] = n
            e["id"] = f"BugEvent:{rec.product}:{mid}:{n}"
    linked = [e for e in all_events if e["bug_id"]]
    for e in all_events:
        if not e["bug_id"]:
            rep["unmatched_events"].append({"message_id": e["message_id"], "type": e["type"],
                                            "mention": e.get("mention"), "channel_id": e["channel_id"],
                                            "ts": e["ts"]})
    ordered = sorted(linked, key=lambda e: (_dt(e["ts"]), e["msg_pos"], e["n"]))
    for seq, e in enumerate(ordered):
        e["seq"] = seq

    def person(eid):
        if eid in people:
            return f"Person:{eid}"
        people.add(eid)
        emp = rec.employees.get(eid)
        nodes[f"Person:{eid}"] = Node("Person", f"Person:{eid}", {
            "eid": eid, "name": emp["name"] if emp else None, "role": emp["role"] if emp else None,
            "in_directory": emp is not None})
        return f"Person:{eid}"

    per_bug: dict[str, list[dict]] = defaultdict(list)
    for e in ordered:
        per_bug[e["bug_id"]].append(e)
    resolved_no_pr = 0
    events_by_type: Counter = Counter()
    for bug_id, b in all_bugs.items():
        evs = per_bug.get(bug_id, [])
        fix_prs = []
        for e in evs:
            for link in e["fix_pr_links"]:
                if cat.pr(link):
                    if link not in fix_prs:
                        fix_prs.append(link)
                elif link not in rep["fix_links_not_in_catalogue"]:
                    rep["fix_links_not_in_catalogue"].append(link)
            if e["type"] == "RESOLVED" and not any(cat.pr(link) for link in e["fix_pr_links"]):
                resolved_no_pr += 1
        derived = derive_bug(evs, fix_prs)
        companies = sorted({(rec.customers.get(c) or {}).get("company") for c in b["customer_ids"]} - {None})
        nodes[bug_id] = Node("Bug", bug_id, {
            "id": bug_id, "title": b["title"], "aliases": b["aliases"], "component": b["component"],
            "impact_text": b["impact_text"], "customer_ids": b["customer_ids"], "companies": companies,
            "channel_id": b["channel_id"], "channel_name": b["channel_name"], "first_reported": b["first_ts"],
            "product": rec.product, **derived})
        edge("HAS", f"Product:{rec.product}", bug_id, [b["message_id"]])
        for cid in b["customer_ids"]:
            c = rec.customers.get(cid, {})
            nodes[f"Customer:{cid}"] = Node("Customer", f"Customer:{cid}", {
                "id": cid, "company": c.get("company"), "name": c.get("name"), "role": c.get("role")})
            customers_seen.add(cid)
            edge("REPORTED_BY", bug_id, f"Customer:{cid}", [b["message_id"]])
        for link in fix_prs:
            ids = [e["message_id"] for e in evs if link in e["fix_pr_links"]]
            edge("FIXES", f"PR:{link}", bug_id, ids)
        for e in evs:
            events_by_type[e["type"]] += 1
            m = by_id[e["message_id"]]
            nodes[e["id"]] = Node("BugEvent", e["id"], {
                "id": e["id"], "type": e["type"], "ts": e["ts"], "message_id": e["message_id"],
                "channel_id": e["channel_id"], "bug": bug_id, "seq": e["seq"], "assignee_eids": e["assignee_eids"],
                "fix_pr_links": e["fix_pr_links"], "citations": e["citations"], "source": e["source"],
                "link_method": e["link_method"], "product": rec.product})
            edge("HAS_EVENT", bug_id, e["id"], [e["message_id"]])
            if not m.is_bot:
                edge("ACTED_BY", e["id"], person(m.user_id), [e["message_id"]])
            for a in e["assignee_eids"]:
                edge("ASSIGNED_TO", e["id"], person(a), [e["message_id"]])
    for x in all_externals:
        xid = f"ExternalBug:{x['link']}"
        nodes[xid] = Node("ExternalBug", xid, {"link": x["link"], "title": x["title"], "project": x["project"],
                                                "similar_mention": x["similar_mention"], "message_id": x["message_id"],
                                                "citations": x["citations"], "source": x["source"]})
        if x["bug_id"]:
            edge("SIMILAR_TO", xid, x["bug_id"], [x["message_id"]])
        else:
            rep["unlinked_external"].append({"link": x["link"], "message_id": x["message_id"]})

    per_company: Counter = Counter()
    for b in all_bugs.values():
        for company in {(rec.customers.get(c) or {}).get("company") for c in b["customer_ids"]} - {None}:
            per_company[company] += 1
    report = {
        "product": rec.product, "prompt_version": PROMPT_VERSION, "rules_version": br.RULES_VERSION,
        "bugs": len(all_bugs), "events_by_type": dict(sorted(events_by_type.items())),
        "events": {"linked": len(linked), "unmatched": len(all_events) - len(linked)},
        "unmatched_events": rep["unmatched_events"],
        "external_look_alikes": len(all_externals), "unlinked_external": rep["unlinked_external"],
        "bugs_per_company": dict(sorted(per_company.items())),
        "customers_without_directory_entry": sorted(c for c in customers_seen if c not in rec.customers),
        "resolved_without_catalogued_pr": resolved_no_pr,
        "fix_links_not_in_catalogue": rep["fix_links_not_in_catalogue"],
        "sessions": rep["sessions"], "model_sessions": rep["model_sessions"],
        "residual_messages": rep["residual_messages"],
        "calls": {"session": caller.calls["session"], "link": caller.calls["link"],
                  "cached": sum(caller.cached.values())},
        "usage": {"prompt_tokens": caller.prompt_tokens, "completion_tokens": caller.completion_tokens,
                  "cost_usd": round(caller.cost_usd, 6)},
        "extraction_failures": rep["extraction_failures"],
        "dropped": {k: dict(v) for k, v in rep["dropped"].items()},
        "rules_dropped": dict(rep["rules_dropped"]), "repaired_citations": rep["repaired_citations"],
        "split_sessions": rep["model_calls_split_sessions"],
        "ignored_model_events": rep["ignored_model_events"],
        "assignment_placement": {"moved_to_handoff": rep["placement"]["moved_to_handoff"],
                                 "duplicate_dropped": rep["placement"]["duplicate_dropped"]},
        "estimate": {"prompt_tokens": caller.est_in, "completion_tokens": caller.est_out,
                     "projected_usd": round(caller.projected_usd(), 6)} if dry else None,
    }
    facts = Facts([], []).merge(Facts(list(nodes.values()), edges))
    return BugResult(facts, report)


def estimate_bugs(rec, *, cache_root=None) -> dict:
    """Calls a live run would make now (cache misses), without sending anything. The channel linking calls depend on
    what the session calls return, so `link_calls_max` is the count when only the rules' results are known."""
    res = run_bugs(rec, chat=None, cache_root=cache_root, dry=True)
    r = res.report
    return {"session_calls": r["calls"]["session"], "link_calls_max": r["calls"]["link"], "cached": r["calls"]["cached"],
            "projected_usd": r["estimate"]["projected_usd"], "estimate": r["estimate"]}
