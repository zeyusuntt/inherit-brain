"""Versioned prompt constants for the semantic layer (Task 4c). Change a prompt, bump the version: the response cache is
keyed by version and by the exact request, so an old answer is never reused for a new prompt."""

SEMANTIC_PROMPT_VERSION = "semantic-v1"

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

"features": a feature or work area the team names as something to build or change ("real-time alerts", "predictive \
analytics integration"). Fields: local_id (a short string such as "f1"), title (a short noun phrase as the speakers \
say it; when the topic is already in KNOWN FEATURES, copy that title exactly), details (specific capabilities, \
technologies or design points proposed for it, each a short phrase from the speech; [] if none), citations.

"contributions": a person's part in a feature. Fields: feature_ref (a local_id), segment_ref (the segment where THAT \
PERSON speaks: the one who accepts, offers or proposes, never the one who asks), kind, citations. kind: "lead" (asked \
to lead, or says they will lead), "assigned" (confirms they are assigned the task or takes it: "I'll handle this"), \
"design" (proposes a concrete technical or design point), "support" (offers help, review or guidance).

"development_links": a PR named in the meeting that belongs to a feature. Fields: feature_ref, pr (the PR title \
copied exactly from the PR LIST, or a PR link), citations. Only PRs that the speakers name; never guess.

"feature_events": what happens to a feature. Fields: feature_ref, kind, reason, pr (an exact PR title, or null), \
segment_ref, citations. kind: "discussed" (the feature is on the agenda; once per feature), "implemented" (a speaker \
says the work or PR is complete, merged or shipped), "held" (a decision to delay, drop or not merge it), "resumed" \
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

"reviews": a person proposes a change to a planning document (Market Research Report, Product Vision Document, \
Product Requirements Document, Technical Specifications Document, System Design Document), or asks a question about \
it. One item per remark. Fields: message_id, document_ref (the document id or link from the DOCUMENTS list, or the \
document type as written, or null), kind ("suggestion" or "review_question"), section (as the author names it, or \
null), text (the remark in one sentence, in the author's words), citations. Thanks, agreement, the document owner's \
replies and summaries of the changes are not reviews.

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
  A product to compare against is not a competitor: only products the messages call competitors, rivals or \
alternatives. Opinions of colleagues about what to do are not claims.

"pr_decisions": a person says what should happen to a pull request. Fields: message_id, prs (the PR numbers or links \
as written, for example ["8"] or ["https://github.com/..."]; all PRs the message is about), action, reason, \
conditional, citations.
  action: "withhold" (an instruction or conclusion not to merge it or to hold it), "resume" (go ahead again after a \
hold), "merge" (an instruction or conclusion to merge it), "revert" (an instruction to revert it), "question" (only \
asking whether to merge, hold or revert, or voicing a tentative preference).
  reason: "security", "product_direction", "introduced_bugs" or "other"; null when none is given.
  conditional: true when the request depends on a condition ("if it is already merged, revert it").
  Report the deciding or instructing message, not every message that agrees.

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
summary, and the feature's title and details.

Answer with one JSON object:
{"assignments":[{"pr":"P1","feature":"F2","quote":"<words copied exactly from that PR's title or summary>"}, \
{"pr":"P2","feature":null,"quote":""}]}

Rules: choose a feature only when the PR clearly develops it; prefer null over a guess. The quote is at most 15 words \
copied character for character from the PR's title or summary and shows why. Every PR gets one entry. Never use a ref \
that is not listed."""
