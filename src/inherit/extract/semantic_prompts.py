"""Versioned prompt constants for the semantic layer (Task 4c). Change a prompt, bump the version: the response cache is
keyed by version and by the exact request, so an old answer is never reused for a new prompt."""

SEMANTIC_PROMPT_VERSION = "semantic-v2"

_CITATIONS = """Every item needs "citations": a list of {"record_id": <id in square brackets>, "quote": <text copied \
exactly from that record>}. A quote is a verbatim span copied character for character from one record, at most 15 \
words, inside a single sentence, including any markdown or emoji exactly as written. Do not invent ids, titles, \
links or people. If a record does not state a fact, report nothing for it."""

MEETING_SYSTEM = """You read one product meeting (numbered transcript segments, plus meeting chat lines when there \
are any) and report facts as JSON.

Two kinds of meeting occur. A DOCUMENT REVIEW meeting discusses one planning document (for example the Market Research \
Report): people suggest changes or ask questions about it. A PRODUCT PLANNING meeting plans or reviews features: people \
propose features, take tasks and report pull requests (PRs).

Return one JSON object with five arrays (use [] when nothing applies):

"document_reviews": a change someone proposes to the document, or a question they ask about it, one item per remark. \
Fields: segment_ref (the id of the segment where that person speaks), kind ("suggestion" or "review_question"), \
section (the document section the remark is about, as the speaker names it, or null), text (the remark in one \
sentence, using the speaker's words), citations. Thanks, agreement and the author's own replies are not reviews.

"features": a feature or work area the team names as a task to build or change ("real-time alerts", "predictive \
analytics integration"). Fields: local_id (a short string such as "f1"), title (a short noun phrase as the speakers \
say it), details (specific capabilities, technologies or design points proposed for it, each a short phrase from the \
speech; [] if none), citations. A pull request title, one implementation step or one technology is not a feature: it \
is a detail of the task it belongs to, or a development link. When the topic is already in KNOWN FEATURES, copy that \
title exactly and do not invent a second feature for it. List only features the speakers talk about in THIS meeting, \
each with at least one citation from it; do not copy the whole KNOWN FEATURES list.

"contributions": a person's part in a feature. Fields: feature_ref (a local_id), segment_ref (the segment where THAT \
PERSON speaks: the one who accepts, offers or proposes), kind, citations. Never put an item on a segment where someone \
asks, requests or assigns ("Ben, can you lead it?"): the item belongs to the segment where Ben answers. kind: "lead" \
(says they will lead, or accepts being asked to lead), "assigned" (confirms they are assigned the task or takes it: \
"I'll handle this", "I confirm"), "design" (proposes a concrete technical or design point), "support" (offers help, \
review or guidance).

"development_links": a pull request that the speakers name by its title or link, for a feature. Fields: feature_ref, \
pr (the PR title exactly as a speaker says it, without the quotation marks, or a PR link), citations. Only PRs that \
a speaker names in this meeting; a feature that only has a plan has none.

"feature_events": what happens to a feature. Fields: feature_ref, kind, reason, pr (an exact PR title, or null), \
segment_ref, citations. kind: "discussed" (the feature is on the agenda; once per feature), "implemented" (a speaker \
says the work or PR is already complete, merged or shipped; plans and promises are not), "held" (a decision to delay, drop or not merge it), "resumed" \
(a decision to go ahead again), "reverted" (said to be rolled back). reason (only for held and reverted): "security", \
"product_direction", "introduced_bugs" or "other"; null otherwise.

""" + _CITATIONS + """

Example (synthetic). Segments: [m_1#4] Ann: "Let's plan the export feature. Ben, can you lead it?" [m_1#5] Ben: \
"Yes, I'll lead the export feature. We could stream rows as CSV." [m_1#6] Cy: "I can help with testing." Output:
{"document_reviews":[],"features":[{"local_id":"f1","title":"export feature","details":["stream rows as CSV"],\
"citations":[{"record_id":"m_1#5","quote":"stream rows as CSV"}]}],"contributions":[{"feature_ref":"f1",\
"segment_ref":"m_1#5","kind":"lead","citations":[{"record_id":"m_1#5","quote":"I'll lead the export feature"}]},\
{"feature_ref":"f1","segment_ref":"m_1#5","kind":"design","citations":[{"record_id":"m_1#5","quote":"stream rows as \
CSV"}]},{"feature_ref":"f1","segment_ref":"m_1#6","kind":"support","citations":[{"record_id":"m_1#6","quote":"I can \
help with testing"}]}],"development_links":[],"feature_events":[{"feature_ref":"f1","kind":"discussed","reason":null,\
"pr":null,"segment_ref":"m_1#4","citations":[{"record_id":"m_1#4","quote":"plan the export feature"}]}]}"""

SESSION_SYSTEM = """You read one Slack session from a product's planning or development channel and report facts as \
JSON. The messages are numbered with their ids; the author of each message is shown.

Return one JSON object with three arrays (use [] when nothing applies):

"reviews": a person proposes a change to the content of a planning document (Market Research Report, Product Vision \
Document, Product Requirements Document, Technical Specifications Document, System Design Document) that is shared or \
named in the session, or asks a question about it. One item per remark. Fields: message_id, document_ref (the document \
id or link from the DOCUMENTS list, the document type as written, or null), kind ("suggestion" or "review_question"), \
section (as the author names it, or null), text (the remark in one sentence, in the author's words), citations. \
Thanks, agreement, the document owner's replies, summaries of the changes, plans to "document the issues somewhere" \
and discussions of what to build are not reviews. Brainstorming who drafts which section is not a review.

"claims": a statement of fact about a competitor, a customer or QA. Fields: message_id (the message that states it; \
its author is the contributor), kind, subject_ref, text (the claim in one sentence, in the message's words), \
customer_ids (CUST-nnnn ids written in the session that the claim is about, [] if none), release_scope, citations.
  kind: "competitor_feature" (a capability of a competitor product; one claim per distinct capability), \
"competitor_weakness" (a weakness or drawback of a competitor; one claim per weakness), "customer_issue" (a customer \
reports a problem with the product), "customer_strength" (a customer praises something about the product), \
"qa_issue" (the QA team found a problem).
  subject_ref: the competitor's name as written (competitor kinds), or the product's name (other kinds).
  release_scope: "previous" when the session says the claim concerns the previous or last release (or the previous \
team's work); "current" when it concerns the product now being built; otherwise "unknown".
  customer_issue and customer_strength need a customer: the cited message itself states what customer CUST-nnnn \
reported or praised. Guesses about causes, ideas for fixes and opinions of colleagues are not claims. A competitor is a \
company's product that the messages call a competitor or rival (see KNOWN COMPETITORS); open-source libraries and \
projects that appear in pull request discussions are not competitors. A message that only shares a demo link makes no \
claim.

"pr_decisions": people decide, or are asked to decide, what happens to a pull request that is approved but not merged \
(whether to merge, hold or revert it, often because priorities, product direction, security or bugs changed). Fields: \
message_id, prs (the PR numbers or links as written, for example ["8"] or ["https://github.com/..."]; all PRs the \
message is about), action, reason, conditional, citations.
  action: "withhold" (an instruction or conclusion not to merge it, or to hold it), "resume" (go ahead again after a \
hold), "merge" (an instruction or conclusion to merge it, after the question or hold was raised), "revert" (an \
instruction to revert it), "question" (only asking whether to merge, hold or revert, or voicing a tentative preference).
  reason: "security", "product_direction", "introduced_bugs" or "other"; null when none is given.
  conditional: true when the request depends on a condition ("if it is already merged, revert it").
  Report the deciding or instructing message, not every message that agrees. Ordinary code review is not a decision: \
"please check my PR", "LGTM, approved", a requested change, or an author saying they will merge their own approved PR \
produce no item.

""" + _CITATIONS + """

Example (synthetic). Messages: [s1] Ann (eid_a): "I was reading about RivalCo, a competitor. It has live dashboards \
and limited export options." [s2] Ben (eid_b): "PR#7 is approved but not merged. Because the roadmap changed, \
please do not merge it." Output:
{"reviews":[],"claims":[{"message_id":"s1","kind":"competitor_feature","subject_ref":"RivalCo","text":"RivalCo has \
live dashboards","customer_ids":[],"release_scope":"unknown","citations":[{"record_id":"s1","quote":"live \
dashboards"}]},{"message_id":"s1","kind":"competitor_weakness","subject_ref":"RivalCo","text":"RivalCo has limited \
export options","customer_ids":[],"release_scope":"unknown","citations":[{"record_id":"s1","quote":"limited export \
options"}]}],"pr_decisions":[{"message_id":"s2","prs":["7"],"action":"withhold","reason":"product_direction",\
"conditional":false,"citations":[{"record_id":"s2","quote":"please do not merge it"}]}]}"""

PR_MEMBERSHIP_SYSTEM = """PR_MEMBERSHIP_TASK. A product team builds features. For each pull request (PR) below, \
choose the one feature in the FEATURES list that the PR develops, or null when none fits. Use the PR's title and \
summary, and the feature's title and details. EXAMPLES lists PRs already assigned from the meetings.

Answer with one JSON object with exactly one entry per PR, in the order listed:
{"assignments":[{"pr":"P1","feature":"F2","quote":"<words copied exactly from that PR's title or summary>"}, \
{"pr":"P2","feature":null,"quote":""}]}

Rules: the PR's own subject decides: a PR about training a forecasting model develops "predictive analytics" even \
when its title never says so. Choose null only when no feature fits. The quote is at most 15 words copied character \
for character from the PR's title or summary and shows why. Never use a ref that is not listed."""

FEATURE_GROUP_SYSTEM = """FEATURE_GROUP_TASK. A product team's meetings produced the feature names below, some of \
them the same work area under different names, or a specific task that belongs under a broader area (for example \
"TLS upgrade" under "security measures", "Kafka consumer group" under "data ingestion efficiency").

Group the features. Answer with one JSON object:
{"groups":[{"title":"F3","members":["F3","F7"]},{"title":"F1","members":["F1"]}]}

Rules: "title" is the ref of the broadest member, whose name becomes the group's name. Every feature is in exactly one \
group. Merge only when the members are the same area or one is a step of the other; keep different areas apart \
(real-time alerts and predictive analytics stay separate). Never use a ref that is not listed."""

DOCUMENT_SYSTEM = """You read one planning document of a product (market research, vision, requirements, technical \
specification or system design) and report facts as JSON.

Return one JSON object with two arrays (use [] when nothing applies):

"features": a product feature or capability the document proposes to build or offer. Fields: title (a short noun \
phrase), details (specific capabilities named for it, short phrases; [] if none), citations. Market segments, \
competitors, risks and general goals are not features.

"claims": a statement about a named competitor product. Fields: kind ("competitor_feature" for a capability, \
"competitor_weakness" for a weakness), subject_ref (the competitor's name as written), text (one sentence), citations. \
Only products the document calls competitors or lists in a competitor comparison.

Every item needs "citations": a list of {"record_id": <the DOCUMENT id>, "quote": <text copied exactly from the \
content>}. A quote is a verbatim span, at most 15 words, inside one sentence. Do not invent features, names or text."""
