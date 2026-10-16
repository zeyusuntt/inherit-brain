"""Rules pass over one bug-channel session (Task 4b). No model calls.

Handles the templated text: quoted bug titles after intake cues, CUST ids, @mention and "can you take this one"
assignments, the responder's self-assignment ("I'll start looking into"), "has been resolved ... PR here: <link>",
"I looked into the reported issue ... not a bug" dismissals, "reopened", and open-source look-alikes. A message that
looks like a lifecycle message but matches no template is listed in `residual` for the model; nothing is guessed.
Release scope and other layers are not touched here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inherit.data.records import parse_mentions
from inherit.extract.catalog import PR_LINK_RE, repo_of, repo_scope
from inherit.extract.cite import _QUOTES
from inherit.extract.structural import CUST_RE, _links

RULES_VERSION = "bug_rules/v1"

# 'Title', "Title" or **Title** (markdown bold). Group 2 holds quoted text, group 3 the bold text; use `qtext`.
QUOTE_RE = re.compile(r"""(?<!\w)(['"])(?=\S)(.{3,200}?)(?<=\S)\1(?!\w)|\*\*(?=\S)(.{3,200}?)(?<=\S)\*\*""", re.S)
SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9'\"])")
INTAKE_OPENER = re.compile(r"\bcustomers?\b.{0,30}\breport(?:ing|ed)\b|\breported by customers\b", re.I | re.S)
INTAKE_CUE = re.compile(r"\b(?:first up|next|lastly|finally|we have|there'?s an?|there is an?)\b", re.I)
BUG_WORD = re.compile(r"\b(?:issue|bug|problem|vulnerability|leak)\b", re.I)
TITLED = re.compile(r"\btitled\b", re.I)
ASSIGN_CUE = re.compile(
    r"\b(?:can you|could you|would you|will you|please|this one'?s (?:for|also for) you|this one is (?:also )?for you"
    r"|take (?:this|it|care)|handle|assign|over to you|look into|have a look|pick (?:this|it) up|for you\b"
    r"|you'?re (?:also |up|assigned|on )|also assigned|up again)", re.I)
ACK_START = re.compile(r"^\W*(?:got it|sure|absolutely|on it|will do|no problem|understood|certainly|of course|yes|"
                       r"i'?ll|i will|i can|i'?m on)\b", re.I)
ACK_BODY = re.compile(r"\b(?:look|dive|tackle|investigat|work|start|take|handle|prioriti[sz]e|fix|address|check|"
                      r"dig)\w*\b", re.I)
RESOLVED_AUX = re.compile(r"\b(?:has|have|had|is|are|was|now) (?:been |now )?(?:resolved|fixed|closed)\b", re.I)
REOPEN = re.compile(r"\bre-?opened\b", re.I)
NOFIX = re.compile(
    r"not an? (?:actual |real )?(?:bug|issue|defect)|does not require (?:a |any )?(?:fix|change)|"
    r"doesn'?t require (?:a |any )?(?:fix|change)|no fix (?:is )?(?:needed|required)|no changes? (?:are |is )?"
    r"(?:needed|necessary|required)|expected behaviou?r|(?:working|functioning|behaving) as (?:intended|designed|expected)|"
    r"by design|won'?t fix", re.I)
GENERIC_CUE = re.compile(RESOLVED_AUX.pattern + "|" + REOPEN.pattern + "|" + NOFIX.pattern + r"|\bre-?assign\w*|"
                         r"duplicate of", re.I)
LOOKED_INTO = re.compile(r"\blooked into the (?:reported )?(?:issue|bug)\s+(?:about|regarding|with|on|where)\s+(.+)",
                         re.I | re.S)
ABOUT_RES = re.compile(r"\b(?:the|this|that) (?:bug|issue)\s+(?:about|regarding|with|related to)\s+(.+?),?\s+"
                       r"(?:has|have|had) been (?:resolved|fixed)\b", re.I | re.S)
OUR_QUOTE = re.compile(r"""\bour\s+(['"])(.{3,200}?)\1(?!\w)""", re.I | re.S)
_CAP = r"[A-Z][A-Za-z0-9\-]*(?:\.[A-Za-z0-9]+)*"
COMPONENT_RE = re.compile(rf"({_CAP}(?: {_CAP}){{0,3}}) & ({_CAP}(?: {_CAP}){{0,3}})")
_STOP_LEAD = frozenset({"This", "That", "The", "Our", "Users", "It", "These", "Those", "In", "On", "At", "And",
                        "Also", "Both", "Which", "Customers", "Admins"})


def straight(text: str) -> str:
    """Curly quotes to straight, same length, so offsets map back onto the original text."""
    return (text or "").translate(_QUOTES)


def norm_title(text: str, names=()) -> str:
    """Matching key: straight quotes, case-folded alphanumeric words; a leading article and a trailing ` in <product or
    alias>` are dropped. Used to compare mentions with titles and aliases, never for bug ids."""
    t = " ".join(re.findall(r"[a-z0-9]+", straight(text).casefold()))
    t = re.sub(r"^(?:the|a|an|our) ", "", t)
    for n in sorted((" ".join(re.findall(r"[a-z0-9]+", x.casefold())) for x in names), key=len, reverse=True):
        if n and t.endswith(" in " + n):
            t = t[: -len(n) - 4]
            break
        if n and t.endswith(" in the " + n + " system"):
            t = t[: -len(n) - 15]
            break
    return t


def slug(title: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", straight(title).casefold()))


AFFECTS_RE = re.compile(rf"\b(?:affect(?:s|ing)?|impact(?:s|ing)?)\s+(?:the |our )?({_CAP}(?: {_CAP}){{0,4}})")
AREA_IS_RE = re.compile(rf"\baffected area is (?:the |our )?({_CAP}(?: {_CAP}){{0,4}})")
IN_AREA_RE = re.compile(rf"\bin (?:the |our )({_CAP}(?: {_CAP}){{0,4}}) area\b")
_NOT_COMPONENT = frozenset({"Customer", "Customers", "CUST", "Users", "Admins", "Sales", "Impacted", "Both", "Several"})


def component_of(text: str) -> str | None:
    """The product area a bug intro names: `A B & C D` first, else the capitalised phrase after affects/impacts
    (`This affects API Security and ...`). Never a customer phrase. None when the text names no capitalised area."""
    t = straight(text)
    for m in COMPONENT_RE.finditer(t):
        left = m.group(1).split(" ")
        while left and left[0] in _STOP_LEAD:
            left = left[1:]
        if left:
            return " ".join(left) + " & " + m.group(2)
    for rx in (AREA_IS_RE, IN_AREA_RE):
        m = rx.search(t)
        if m:
            return m.group(1)
    for m in AFFECTS_RE.finditer(t):
        words = m.group(1).split(" ")
        if words[0] in _NOT_COMPONENT or words[0].startswith("CUST"):
            continue
        return " ".join(words)
    return None


def sentence_span(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for b in SENT_BOUNDARY.finditer(text):
        if b.end() <= pos:
            start = b.end()
        elif b.start() >= pos:
            return start, b.start()
    return start, len(text)


def qtext(m) -> str:
    return m.group(2) or m.group(3)


def assignees_in(text: str, author: str) -> list[str]:
    """Employee ids @mentioned in the sentences that carry an assignment cue. A mention elsewhere ("Thanks, @eid_x!"
    thanking the previous assignee) is not an assignee."""
    out: list[str] = []
    start = 0
    spans = []
    for b in SENT_BOUNDARY.finditer(text):
        spans.append((start, b.start()))
        start = b.end()
    spans.append((start, len(text)))
    for a, b in spans:
        sent = text[a:b]
        if ASSIGN_CUE.search(sent):
            out += [x for x in parse_mentions(sent) if x != author and x not in out]
    return out


def quote_at(text: str, pos: int) -> str:
    a, b = sentence_span(text, pos)
    return text[a:b].strip()


def first_quote(text: str, start: int = 0, end: int | None = None):
    m = QUOTE_RE.search(straight(text), start, len(text) if end is None else end)
    return m


def quotes(text: str) -> list:
    return list(QUOTE_RE.finditer(straight(text)))


def area_key(text: str) -> str:
    """Comparison key of a product area: case-folded words, `&` and `and` equal."""
    return " ".join(w for w in re.findall(r"[a-z0-9]+", straight(text).casefold().replace("&", " and ")))


_FUNCTION_WORDS = frozenset("the a an in of for to during on at and with from by our is are be this that its their".split())


def content_words(text: str) -> frozenset:
    """The words of a bug name without articles and prepositions: `in`, `during the` and the like do not distinguish
    one bug from another."""
    return frozenset(w for w in area_key(text).split() if w not in _FUNCTION_WORDS and w != "and")


def cust_ids(text: str) -> list[str]:
    return list(dict.fromkeys(CUST_RE.findall(text or "")))


def pr_links(text: str) -> list[str]:
    out = []
    for link in sorted(_links(text or ""), key=lambda x: text.find(x)):
        m = PR_LINK_RE.match(link)
        if m:
            out.append(re.match(r"https?://github\.com/[^/]+/[^/]+/pull/\d+", link).group(0))
    return list(dict.fromkeys(out))


@dataclass
class SessionParse:
    bugs: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    externals: list[dict] = field(default_factory=list)
    residual: list[str] = field(default_factory=list)
    handled: set[str] = field(default_factory=set)
    reopen: bool = False
    no_area: list[str] = field(default_factory=list)  # local ids of rules bugs whose text names no product area


def _cite(m, quote: str) -> dict:
    return {"record_id": m.id, "quote": quote}


def parse_session(session, cat, customers) -> SessionParse:
    """`session`: SlackMessages of one bug-channel session in time order. `cat`: the product Catalog. Returns the
    candidates found by rules, the messages they came from and the lifecycle-looking messages left for the model."""
    p = SessionParse()
    msgs = [m for m in session if not m.is_bot]
    if not msgs:
        return p
    opener = msgs[0]
    intake_session = bool(INTAKE_OPENER.search(opener.text))
    looked_into = LOOKED_INTO.search(straight(opener.text))
    claimed: set[str] = set()  # messages whose lifecycle role is settled (rules event, or deliberately not a cue)
    counter = 0
    bug_rows: list[tuple] = []

    def ev(type_, m, mention, cites, local_bug=None, assignees=(), fix=()):
        p.events.append({"type": type_, "message_id": m.id, "mention": mention, "local_bug": local_bug,
                         "assignee_eids": list(assignees), "fix_pr_links": list(fix), "citations": cites,
                         "source": "rules"})
        p.handled.add(m.id)

    def fix_links_from(i: int) -> tuple[list[str], list[dict]]:
        author, links, cites = msgs[i].user_id, [], []
        for j in range(i, len(msgs)):
            m = msgs[j]
            if m.user_id != author or (j > i and RESOLVED_AUX.search(m.text)):
                if j > i and m.user_id == author:
                    break
                continue
            found = [x for x in pr_links(m.text) if repo_scope(x, cat.product_repos) == "product"]
            if found:
                cites.append(_cite(m, found[0]))
                links += [x for x in found if x not in links]
        return links, cites

    intake_idx = [i for i, m in enumerate(msgs) if m.user_id == opener.user_id and not RESOLVED_AUX.search(m.text)
                  and not REOPEN.search(m.text)]
    for i, m in enumerate(msgs):
        text = m.text
        ext = [x for x in pr_links(text) if repo_scope(x, cat.product_repos) == "external"]
        # ------------------------------------------------ open-source look-alikes
        if ext and not RESOLVED_AUX.search(text):
            qs = quotes(text)
            if len(ext) != 1 or not qs:
                p.residual.append(m.id)
                continue
            title = qtext(qs[0])
            ours = OUR_QUOTE.search(straight(text))
            similar = ours.group(2) if ours else (qtext(qs[1]) if len(qs) > 1 else None)
            owner_repo = "/".join(repo_of(ext[0]))
            cites = [_cite(m, title), _cite(m, ext[0])]
            if similar:
                cites.append(_cite(m, similar))
            p.externals.append({"link": ext[0], "title": title, "project": owner_repo, "similar_mention": similar,
                                "message_id": m.id, "citations": cites, "source": "rules"})
            p.handled.add(m.id)
            claimed.add(m.id)
            continue
        # ------------------------------------------------ resolved
        sres = RESOLVED_AUX.search(text)
        if sres and re.search(r"\b(?:has|have|had) been (?:resolved|fixed)\b", text, re.I) and not looked_into:
            a, b = sentence_span(text, sres.start())
            sent = text[a:b]
            mention = None
            qm = first_quote(sent)
            about = ABOUT_RES.search(straight(sent))
            if qm:
                mention = qtext(qm)
            elif about:
                mention = text[a:b][about.start(1):about.end(1)]
            if mention:
                links, lcites = fix_links_from(i)
                ev("RESOLVED", m, mention, [_cite(m, sent.strip()), *lcites], fix=links)
                claimed.add(m.id)
                for j in range(i + 1, len(msgs)):  # the same author's link messages belong to this event
                    if msgs[j].user_id == m.user_id and any(c["record_id"] == msgs[j].id for c in lcites):
                        p.handled.add(msgs[j].id)
                continue
            if i == 0 or m.user_id == opener.user_id:
                p.residual.append(m.id)
            claimed.add(m.id)
            continue
        # ------------------------------------------------ reopened
        sre = REOPEN.search(text)
        if sre:
            p.reopen = True
            a, b = sentence_span(text, sre.start())
            sent = text[a:b]
            qm = first_quote(sent)
            if qm:
                ev("REOPENED", m, qtext(qm), [_cite(m, sent.strip())])
            elif i == 0 or m.user_id == opener.user_id:
                p.residual.append(m.id)
            claimed.add(m.id)
            continue
        # ------------------------------------------------ intake
        qs = quotes(text)
        intake_like = (
            (intake_session and i in intake_idx and i > 0 and (cust_ids(text) or (INTAKE_CUE.search(text) and BUG_WORD.search(text))))
            or (TITLED.search(text) and cust_ids(text) and not looked_into))
        if intake_like:
            titles = list(dict.fromkeys(qtext(q) for q in qs))
            if len(titles) != 1:
                p.residual.append(m.id)
                claimed.add(m.id)
                continue
            counter += 1
            title = titles[0]
            lid = f"rb{counter}"
            cids = cust_ids(text)
            mentions = [x for x in dict.fromkeys(m.mentions) if x != m.user_id]
            cites = [_cite(m, title)] + [_cite(m, c) for c in cids]
            p.bugs.append({"local_id": lid, "title": title, "aliases": [], "component": component_of(text),
                           "impact_text": text.strip(), "customer_ids": cids, "message_id": m.id,
                           "citations": cites, "source": "rules"})
            p.handled.add(m.id)
            claimed.add(m.id)
            bug_rows.append((i, lid, title, m))
            continue
        if intake_session and i in intake_idx and i > 0 and not looked_into and INTAKE_CUE.search(text) \
                and BUG_WORD.search(text):
            p.residual.append(m.id)
            claimed.add(m.id)

    # ---------------------------------------------------------------- assignment of each introduced bug
    bug_idx = [r[0] for r in bug_rows]
    for n, (i, lid, title, m) in enumerate(bug_rows):
        text = m.text
        nxt = next((k for k in bug_idx if k > i), len(msgs))
        own = assignees_in(text, m.user_id)
        if own:
            ev("ASSIGNED", m, title, [_cite(m, quote_at(text, text.find(f"@{own[0]}")))], local_bug=lid, assignees=own)
            continue
        if parse_mentions(text) and ASSIGN_CUE.search(text):  # a cue and a mention that do not sit together
            p.residual.append(m.id)
            continue
        for k in range(i + 1, nxt):  # the lead assigns in a follow-up message
            f = msgs[k]
            got = assignees_in(f.text, m.user_id) if f.user_id == m.user_id and k in intake_idx else []
            if got:
                ev("ASSIGNED", f, title, [_cite(f, quote_at(f.text, f.text.find(f"@{got[0]}")))], local_bug=lid,
                   assignees=got)
                break
        else:
            for k in range(i + 1, nxt):  # no mention: the first responder takes it
                r = msgs[k]
                if r.user_id != m.user_id and ACK_START.search(r.text) and ACK_BODY.search(r.text):
                    ev("ASSIGNED", r, title, [_cite(r, quote_at(r.text, 0))], local_bug=lid, assignees=[r.user_id])
                    claimed.add(r.id)
                    break

    p.no_area = [b["local_id"] for b in p.bugs if not b["component"]]

    # ---------------------------------------------------------------- dismissal: "I looked into the reported issue"
    if looked_into:
        sent = looked_into.group(1)
        end = SENT_BOUNDARY.search(sent)
        first = sent[: end.start()] if end else sent
        qm = QUOTE_RE.search(straight(first))
        mention = qtext(qm) if qm else re.sub(r"^(?:the|a|an)\s+", "", first.strip().rstrip(".!? "), flags=re.I)
        cue = next((m for m in msgs if m.user_id == opener.user_id and NOFIX.search(m.text)), None)
        if cue is not None and mention:
            cm = NOFIX.search(cue.text)
            ev("DISMISSED", cue, mention, [_cite(opener, mention), _cite(cue, quote_at(cue.text, cm.start()))])
            claimed.add(opener.id)
            claimed.add(cue.id)
        else:
            p.residual.append(opener.id)
            claimed.add(opener.id)

    # ---------------------------------------------------------------- other lifecycle-sounding messages
    if not any(e["type"] in ("RESOLVED", "DISMISSED", "REOPENED") for e in p.events):
        for i, m in enumerate(msgs):
            if m.id in claimed or m.id in p.residual:
                continue
            if (i == 0 or m.user_id == opener.user_id) and GENERIC_CUE.search(m.text):
                p.residual.append(m.id)
    seen = set()
    p.residual = [x for x in p.residual if not (x in seen or seen.add(x))]
    p.residual = [x for x in p.residual if x not in p.handled]
    return p
