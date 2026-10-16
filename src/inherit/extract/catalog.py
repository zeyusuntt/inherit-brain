"""Per-product reference catalogue used to resolve references in the bug (4b) and semantic (4c) layers. No model calls.

Holds the product name and aliases (with the records that show each alias), documents by id, type and link, and PRs
by link and exact title with product-repo versus external membership (R3). Duplicate-title PRs stay separate.
Release scope is not here: an alias is an identity, never a scope (R2). `previous_names` is filled by the rules layer.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from inherit.data.records import ProductRecords

PR_LINK_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/pull/\d+")
CHANNEL_CREATED_RE = re.compile(r"This is the very beginning of the (\S+?) channel")
URL_ALIAS_RE = re.compile(r"/archives/([^/]+)/demo_\d+")
DOC_STEM_RE = re.compile(
    r"^(?:(?:final|latest|new)_)?(?P<stem>.+?)_(?:market_research|product_vision|product_requirements"
    r"|technical_specifications|system_design)(?:_report|_document)?(?:_final)?$")
# The organisation that owns the product repositories in the real bundle (salesforce/<repo>, repo = product or alias).
PRODUCT_ORG = "salesforce"


def doc_stem(doc_id: str) -> str | None:
    """Lower-case product stem of a versioned document id, e.g. `prox_market_research_report_final` -> `prox`."""
    m = DOC_STEM_RE.match(doc_id)
    return m.group("stem") if m else None


def repo_of(link: str) -> tuple[str, str] | None:
    m = PR_LINK_RE.match(link or "")
    return (m.group(1), m.group(2)) if m else None


def repo_scope(link: str, product: str, aliases=()) -> str:
    """`product` when the PR lives in the organisation's repository or in a repo or owner named like the product or one
    of its aliases (alias repos included); otherwise `external` (open-source look-alikes)."""
    r = repo_of(link)
    if r is None:
        return "external"
    owner, repo = r
    names = {product.lower(), *(a.lower() for a in aliases)}
    if owner == PRODUCT_ORG or owner.lower() in names or repo.lower() in names:
        return "product"
    return "external"


def channel_label(name: str) -> str | None:
    """Product-like name inside a channel name: planning-<Name>[-PM], develop-<owner>-<Name>, bug-<owner>-<Name>."""
    if name.startswith("planning-"):
        label = name[len("planning-"):]
        return label[:-3] if label.endswith("-PM") else label or None
    parts = name.split("-", 2)
    if len(parts) == 3 and parts[0] in ("develop", "bug"):
        return parts[2] or None
    return None


def derive_aliases(rec: ProductRecords) -> dict[str, list[str]]:
    """alias (display casing) -> sorted evidence strings `<kind>:<record id>`. Sources: channel names, channel creation
    notices (the channel was renamed later), versioned document ids, and demo link paths. The product itself is not an
    alias; neither is any name that only differs from it by case."""
    product = rec.product.lower()
    ev: dict[str, set[str]] = defaultdict(set)

    def add(name: str | None, kind: str, rid: str) -> None:
        if name and name.lower() != product:
            ev[name.lower()].add(f"{kind}:{rid}")

    for ch in dict.fromkeys(m.channel_name for m in rec.messages):
        add(channel_label(ch), "channel_name", ch)
    for m in rec.messages:
        if m.is_bot:
            c = CHANNEL_CREATED_RE.search(m.text)
            if c:
                add(channel_label(c.group(1)), "channel_creation", m.id)
    for d in rec.documents:
        add(doc_stem(d.id), "document_id", d.id)
    for u in rec.urls:
        for link in u.links:
            m = URL_ALIAS_RE.search(link)
            if m:
                add(m.group(1), "url", u.id)

    # Display casing: the most frequent whole-word surface form in the bundle text, preferring mixed case.
    texts = [m.text for m in rec.messages] + [m.channel_name for m in rec.messages] + [d.content for d in rec.documents]
    texts += [s.text for mt in rec.meetings for s in mt.segments] + [p.title + " " + p.summary for p in rec.prs]
    texts += [link for u in rec.urls for link in u.links] + [x for u in rec.urls for x in u.descriptions]
    out: dict[str, list[str]] = {}
    for key, evidence in ev.items():
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        forms: Counter = Counter()
        for t in texts:
            forms.update(pat.findall(t))
        display = min(forms, key=lambda f: (-forms[f], f == f.lower(), f)) if forms else key
        out[display] = sorted(evidence)
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


@dataclass(frozen=True)
class DocEntry:
    id: str
    type: str
    link: str
    author: str
    date: str
    stem: str | None  # lower-case product stem parsed from the id, e.g. prox


@dataclass(frozen=True)
class PrEntry:
    id: str
    link: str
    number: str
    title: str
    author: str
    owner: str | None
    repo: str | None
    repo_scope: str  # product | external
    merged: bool | None
    state: str


@dataclass(frozen=True)
class Catalog:
    product: str
    aliases: tuple[str, ...]
    alias_evidence: dict
    documents: tuple[DocEntry, ...]
    prs: tuple[PrEntry, ...]
    product_repos: frozenset
    previous_names: tuple[str, ...] = ()
    _docs: dict = field(default_factory=dict, compare=False, repr=False)
    _doc_links: dict = field(default_factory=dict, compare=False, repr=False)
    _prs: dict = field(default_factory=dict, compare=False, repr=False)
    _titles: dict = field(default_factory=dict, compare=False, repr=False)

    def with_previous(self, names) -> Catalog:
        return replace(self, previous_names=tuple(names))

    def doc(self, ref: str) -> DocEntry | None:
        """By document id or document link (a Slack `<link|label>` reference also works)."""
        ref = (ref or "").strip().lstrip("<").rstrip(">").split("|", 1)[0]
        return self._docs.get(ref) or self._doc_links.get(ref)

    def doc_by_link(self, link: str) -> DocEntry | None:
        return self._doc_links.get(link)

    def docs_of_type(self, type_: str) -> list[DocEntry]:
        return [d for d in self.documents if d.type == type_]

    def pr(self, link: str) -> PrEntry | None:
        return self._prs.get(link)

    def prs_by_title(self, title: str, scope: str | None = None) -> list[PrEntry]:
        """Every PR with exactly this title; `scope` product or external narrows it. Never deduplicated."""
        return [p for p in self._titles.get(title, []) if scope is None or p.repo_scope == scope]


def build_catalog(rec: ProductRecords) -> Catalog:
    al = derive_aliases(rec)
    aliases = tuple(al)
    docs = tuple(DocEntry(d.id, d.type, d.link, d.author, d.date, doc_stem(d.id)) for d in rec.documents)
    prs = []
    for p in rec.prs:
        r = repo_of(p.link)
        prs.append(PrEntry(p.id, p.link, p.number, p.title, p.author, r[0] if r else None, r[1] if r else None,
                           repo_scope(p.link, rec.product, aliases), p.merged, p.state))
    titles: dict[str, list[PrEntry]] = defaultdict(list)
    for p in prs:
        titles[p.title].append(p)
    return Catalog(
        product=rec.product, aliases=aliases, alias_evidence=al, documents=docs, prs=tuple(prs),
        product_repos=frozenset(f"{p.owner}/{p.repo}" for p in prs if p.repo_scope == "product" and p.owner),
        _docs={d.id: d for d in docs}, _doc_links={d.link: d for d in docs}, _prs={p.link: p for p in prs},
        _titles=dict(titles))
