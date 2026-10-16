"""Versioned prompt constants for the model extraction layers. Change a prompt, bump its version: the response cache is
keyed by version and by the exact request, so an old answer is never reused for a new prompt."""

BUGS_PROMPT_VERSION = "bugs-v5"

BUG_SESSION_SYSTEM = """You read one Slack session from a company's customer-bug channel and report bug facts as JSON.

People post templated updates: a lead introduces customer bugs, engineers acknowledge, later sessions announce fixes, \
no-fix conclusions or reopenings. A rules pass has already parsed the clear cases and lists them for you. Your job is \
only the rest: messages the rules could not parse, and follow-up facts the rules do not cover.

Return one JSON object with four arrays (use [] when nothing applies):

"bugs": a NEW customer bug introduced in this session that the rules did not capture. Fields: local_id (any short \
string, for example "x1"), title (the bug's name as written in the message, not a paraphrase of yours), aliases ([]), \
component (the product area named in the message, or null), impact_text (the sentence describing the impact), \
customer_ids (CUST-nnnn ids written in the messages), message_id (the message that introduces it), citations.

"events": lifecycle events of a bug. Fields: type, bug_ref, bug_mention, actor_message_id, assignee_eids, \
fix_pr_links, citations.
  type is one of ASSIGNED, RESOLVED, DISMISSED, REOPENED. Never emit REPORTED: one is created for each bug.
  ASSIGNED: someone becomes responsible, including self-assignment ("I'll take it") and reassignment. \
assignee_eids lists the employee ids (eid_...) of the people made responsible. Use only eids that appear in the \
message list; a first name alone is resolved with the speaker list, never invented. Asking a colleague to help or assist ("could you assist me") does not assign the bug to them, and neither does \
their reply agreeing to help. In a session about \
open-source look-alikes, people who say they will review those pull requests are not assigned \
to anything. A person thanked in the same \
message ("Thanks, @eid_x!") for an earlier bug is not assigned to the new one: only the person asked or told to \
handle it.
  RESOLVED: the message says the bug is fixed or resolved. fix_pr_links holds pull request links copied from the \
messages.
  DISMISSED: the message concludes it is not a bug, expected behaviour, by design, or needs no fix.
  REOPENED: the bug is reported as occurring again or reopened.
  actor_message_id is the message that states the event; its author is the actor.
  bug_ref names the bug: use a ref from the rules list (for example "m1" or "rb1") or the local_id of a bug you \
return. If neither exists, set bug_ref to null and put the words the message uses for the bug in bug_mention, copied \
exactly from the message.

"external": an open-source look-alike (a bug in another project, with a link to that project's pull request). Fields: \
title, link, similar_mention (the customer bug it is compared to, copied from the message, or null), message_id, \
citations. Open-source bugs are never entries in "bugs".

"components": the product area a bug is said to affect, only for the bugs the user message lists under BUGS WITHOUT \
A PRODUCT AREA. Fields: bug_ref (the ref from that list), component (the area exactly as the message writes it, \
any case; one of the KNOWN AREAS when the message names it), citations. Skip a bug when its message names no area.

Every item needs "citations": a list of {"record_id": <message id>, "quote": <text copied exactly from that message>}. \
Quotes are verbatim spans copied character for character from one message, at most 15 words, inside a single \
sentence, including any markdown such as ** exactly as written. A resolution, dismissal or reopening must be stated in the cited message; \
a promise or plan to look again ("I'll revisit it") is none of them. Do not repeat anything the rules already parsed. Do not invent \
ids, titles, links or people. If a message does not state a bug fact, report nothing for it.

Example (synthetic). Messages: [m1] Lead (eid_aaa): "Next, there is a slow export issue. Customer CUST-0001 is \
affected. Dana, can you take it?" [m2] Dana (eid_bbb): "Yes, I'll start on it." Output:
{"bugs":[{"local_id":"x1","title":"slow export","aliases":[],"component":null,"impact_text":"slow export issue",\
"customer_ids":["CUST-0001"],"message_id":"m1","citations":[{"record_id":"m1","quote":"a slow export issue"}]}],\
"events":[{"type":"ASSIGNED","bug_ref":"x1","bug_mention":null,"actor_message_id":"m2","assignee_eids":["eid_bbb"],\
"fix_pr_links":[],"citations":[{"record_id":"m2","quote":"I'll start on it"}]}],"external":[]}"""

BUG_LINK_SYSTEM = """LINK_TASK. A product's customer bugs were introduced in one Slack channel. Later messages \
resolve, dismiss, reopen or compare bugs, often with different words than the title. For each item, choose the one \
bug it is about, or null.

Input lists BUGS (ref, title, aliases, component, customers, when first reported) and ITEMS (ref, kind, event type, \
the words the message uses for the bug, the message text, the time). Answer with one JSON object:
{"links":[{"item":"I1","bug":"B3"}, {"item":"I2","bug":null}]}

Rules: choose a bug when the message describes the same problem, even in other words (a paraphrase, a shortened \
title, extra words such as the product name; a mention that contains a listed title followed by extra words refers to \
that bug). Do not choose a bug when the meaning or the causal direction differs, \
for example which side overwrites which, even if the words are the same. Answer null when no listed bug is the same \
problem, but never to avoid a decision when one bug clearly fits. A bug cannot be the target of an item that happened before the bug was first reported. Never \
create a bug and never use a ref that is not listed. Several items may point to the same bug. Cover every item."""
