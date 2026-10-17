# Pre-registration: inherit-brain on HERB

Generated 2026-10-17T02:49:34Z from commit 8cc438a70713d6f05d203e377c7d28c9898b6b9e. Nothing below changes after this file's hash is posted. Any later edit is published as a separate, dated "post-hoc version" and never replaces this one.

## 1. What this file fixes

This file fixes, before any held-out question is answered by any arm: the inputs and their hashes, the extracted facts, the code and prompts, the arms and their budgets, the endpoints and hypotheses with their thresholds, the scoring and statistics, the multi-hop family list, and the sentence we will publish for each outcome.

Time evidence: this file is committed to a private repository; its sha256 is published in a short public post, and the post is archived in the Wayback Machine. The held-out run starts only after both exist and are recorded in `results/freeze/release.json`; the run refuses to start otherwise.

## 2. Identity

| Item | Value |
|---|---|
| Code commit | 8cc438a70713d6f05d203e377c7d28c9898b6b9e |
| Upstream HERB commit | db3bf9b3f911745726c579c9dbf9f7f6b2c05b36 |
| Split hash | 70ec2f994d94aeb2854ddbbcb204c3a286aa7e393b583ff9c99c6fa9c2846e60 |
| Input bundle sha256 | 029c8028d2d08bd50d800865a1768d926d719470e66ef9729dd41f23296af091 |
| Freeze manifest sha256 | 61b7da15cb0f0176af156d858720e0a3ccd13d9acec53a09410dfe69f28d9dc9 |
| Effective configuration hash | 42242b0a15016892476a7fc229f23edc1d54763946fbaefa9c81284a5ab9ad28 |
| Acceptance report sha256 | 6a8fea72a6e2f5ea2077f77ecffa60361867b9d73d5ab6dcadacba87ecd2bb34 |

facts hash: dfe2d0b7d57ac98e00e46fe168fcfe9ada7fb0158cc26642cbec1a4cc19d8d02

Versions of every prompt and protocol:

| Item | Version |
|---|---|
| agent_prompt | agent/v5 |
| approval_predicate | pr_approval/v1 |
| bug_rules | bug_rules/v1 |
| bugs_prompt | bugs-v5 |
| card | v2 |
| feature_fate | feature_fate/v1 |
| judge_prompt | herb-evaluate-v1 |
| pending | pending/v1 |
| role_cues | v2 |
| rules.approval | pr_approval/v1 |
| rules.demo | demo/v1 |
| rules.named_qa | named_qa/v1 |
| rules.revert | revert/v1 |
| rules.scope | scope/v1 |
| scorer | eval.grade/2 |
| semantic_prompt | semantic-v2 |
| arm A0 baseline | baselines/a0/v3 |
| arm A0 config sha256 | 80d2c147090ce25b74e1cda1b17de3d47b10afb5935a28093649632fc92582d8 |
| arm A0f baseline | baselines/a0/v3 |
| arm A0f config sha256 | 71c5edb9a937f37e1e354e8c3cd8a83bb4f499e53fdfd9b211b7c06f01c30ed6 |
| arm A0v baseline | baselines/a0/v3 |
| arm A0v config sha256 | 588bd8c9e23a5695347e67239330879d83eaa3c988f25f26c50615493bef4f6f |
| arm A1f baseline | baselines/a1f/v3 |
| arm A1f config sha256 | 45265622a0f262a77a16692a01abaaa35b949d86034d0a193d87c3b407337cbb |
| arm A2 a2 | a2/v4 |
| arm A2 config sha256 | 2b601e0ed28036c0f2dc177dca201748890687730996a953246b41aaf5949c8d |
| arm A4 agent | agent/v5 |
| arm A4 config sha256 | bb4f9e9d7a0f5c1377acb3c539e16418b5a3cdfdb42cafad83a1e54f7e4e142f |
| arm A4c agent | agent/v5 |
| arm A4c config sha256 | 77f75a202bfe1171b9740c422ef07b0ce4c635ba36a67f573ef82cfccc1bfd7d |
| arm A5 agent | agent/v5 |
| arm A5 config sha256 | 8e25352ef8ac3dc3a86955044520a884ff65a613830e4be60001500518f7bb08 |

## 3. Data and split

- HERB (Salesforce AI Research), 30 products. Six products are the development set: ActionGenie, AnomalyForce, ContentForce, KnowledgeForce, LeadForce, TrendForce. The other 24 are held out.
- Held-out questions: 659 answerable and 544 unanswerable. Of the answerable ones, 416 have a normalised question string that also occurs in the development set and 243 do not.
- Inputs allowed into extraction and the graph: each product's slack, documents, meeting_transcripts, meeting_chats, urls and prs, plus metadata/employee.json and metadata/customers_data.json. Excluded: the products' team and customers fields, the questions, ground truth and citations. The bundle is built from the allowed fields only; its hash is above.
- All 30 products were extracted and loaded with the same frozen code before this file was written. No arm answered a held-out question during development, and no held-out answer, gold value or citation was displayed or used. Held-out questions were loaded only by eval-side code, to count them per product (for the budget) and to compute their normalised strings (for section 11).

## 4. Model and limits

- Every arm uses model M, `Qwen/Qwen3-30B-A3B-Instruct-2507` on Nebius, at the settings in the configuration hash. Embeddings: `Qwen3-Embedding-8B`, 1,024 dimensions.
- Every model call goes through one local metering proxy with a 50 USD hard cap for the whole project and a sub-cap per arm (section 7). A call that would cross a cap is refused.
- Agent arms (A5, A4, A4c, A0, A0f) share one loop with the same limits: at most 12 tool calls, 24 model calls, 60,000 cumulative input tokens and 300 seconds per question; 1,024 output tokens per model call. The input budget was raised from 40,000 after a dev pilot in which A4 stopped on it in 22 of 48 questions (pre-registered rule: above a quarter, raise it for every loop arm and pilot again). In the second pilot, at 60,000, A4 stopped in 41 of 48 (it uses whatever budget it gets), A0 in 17 of 48 instead of 33, and no loop arm's pilot score rose. The budget stays at 60,000 because it starves A0, the comparator of H1, less; raising it further would not fit the 50 USD cap.
- A1f, A0v and A2 answer with one model call over their own context. A1f's call holds the whole product (about 110,000 to 140,000 tokens), so the loop's cumulative budget does not bind it; its limit is the model's 262,144-token context.

## 5. The four frozen protocols

1. Bug states. Events are ordered by message time; types are REPORTED, ASSIGNED, RESOLVED (with a fix PR), DISMISSED and REOPENED. "Current" means the last day of the corpus. Unresolved: no linked fix PR. Active: the last event is REPORTED, ASSIGNED or REOPENED. Reopened: at least one REOPENED event. No fix needed: at least one DISMISSED event.
2. Time. Every event takes its message timestamp; the UDF `herb_v1.statusAt(events, date)` gives a bug's state on a date under the rules above.
3. Name resolution. For a name spoken in a meeting, the candidates are the same-name attendees of that meeting; only if none exists are same-name people with a related Slack post within 3 days or a mention considered, and a same-name non-attendee is added (making the name ambiguous) only when the speaker's own words match that person's role and no attendee's role does. Score = 3 x role cue (the spoken role matches the directory role) + 2 x the candidate posted in a related Slack thread within 3 days + 1 x the candidate was @-mentioned in the same meeting. A unique top score resolves the name; a tie keeps every tied candidate, marked ambiguous, and the answer says so.
4. Refusal. An arm answers "not in the records" only when an entity the question names cannot be resolved, or when every relation the question needs returns nothing. Partial results are answered as partial. Every arm gives the same refusal sentence, "I could not find information in the records to answer this question.", with empty answer_ids.

## 6. Entity cards

A card is a node's own fields plus, for each directly connected node, its type, id and name (not its fields), and each edge's source record ids. The A0f retrieval corpus and the A4 tools read the same cards. The complete field list:

<!-- fields:begin -->

### Product

- display name: `name`
- props: `name`, `aliases`
- neighbours: only `HAS` out-edges to Bug, Document, Meeting, Feature, PR, Competitor (`edge`, `dir`, `label`, `id`, `name`; no `src`, no messages, no counts)

### Person

- display name: `name`, else the eid
- props: `eid`, `name`, `role`, `in_directory`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Customer

- display name: `company`
- props: `id`, `company`, `name`, `role`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Channel

- display name: `name`
- props: `name`, `bug_owner`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Message

- display name: the message id
- props: `ts`, `kind`, `user_id`, `text`, `release_scope`
- neighbours: all incident edges, including other Messages' edges to it (the poster, channel, mentioned people, PRs, features, documents)

### Document

- display name: `type` + " (" + document id + ")"
- props: `type`, `date`, `link`, `content`, `feedback`, `feedback_items`, `release_scope`, `source_product_name`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Meeting

- display name: `document_type` + " meeting (" + meeting id + ")"
- props: `document_type`, `date`, `release_scope`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### PR

- display name: `title`
- props: `title`, `number`, `summary`, `state`, `merged`, `mergeable`, `created_at`, `repo`, `repo_scope`, `approval`, `open_unmerged`, `pending_action`, `pending_reason`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Url

- display name: the link
- props: `link`, `descriptions`, `kind`, `release_scope`, `source_product_name`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Bug

- display name: `title`
- props: `title`, `aliases`, `component`, `impact_text`, `first_reported`, `status`, `unresolved`, `active`, `reopened`, `no_fix_needed`, `resolved`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### BugEvent

- display name: the event id
- props: `ts`, `type`, `seq`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Feature

- display name: `title`
- props: `title`, `topic`, `aliases`, `details`, `implemented`, `reverted`, `held`, `delayed_then_added`, `discussed_not_implemented`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### FeatureEvent

- display name: the event id
- props: `kind`, `reason`, `reason_text`, `date`, `record_id`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Competitor

- display name: `name`
- props: `name`, `aliases`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### ExternalBug

- display name: `title`
- props: `title`, `link`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

### Observation

- display name: the observation id
- props: `kind`, `text`, `release_scope`, `record_id`
- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours

<!-- fields:end -->

Choices that bound what a card shows: Message neighbours are omitted from every non-Message card (messages are evidence, not entities); the derived COLLABORATED edge is excluded; at most 3 source record ids are listed per neighbour entry, then "..."; the Product card lists only bugs, documents, meetings, features, PRs and competitors, with no messages and no counts; one depth-1 response returns at most 200 rows and 12,000 tokens, with a cursor for the next page.

## 7. Arms

Every arm answers in the same JSON shape with the same answer instructions (prompt versions above): answer_text, answer_ids, outcome, then at most 10 distinct evidence ids. Only the answer fields are scored. A reply cut off inside the final evidence list is accepted when every graded field before it is complete; graded fields are never repaired.

| Arm | Held-out products | Sub-cap USD |
|---|---|---|
| A0 | all 24 | 7.64 |
| A0f | CollaborateForce, ConnectForce, FeedbackForce, MonitorForce, SecurityForce | 2.36 |
| A0v | all 24 | 0.94 |
| A1f | all 24 | 19.53 |
| A2 | AutoTuneForce, CoachForce, EdgeForce, PitchForce, ProposalForce, WorkFlowGenie | 0.47 |
| A4 | CollaborateForce, ConnectForce, FeedbackForce, MonitorForce, SecurityForce plus every held question in the multi-hop families | 2.42 |
| A4c | CollaborateForce, ConnectForce, FeedbackForce, MonitorForce, SecurityForce | 2.15 |
| A5 | all 24 | 1.76 |

A2 (GraphRAG-SDK 1.4.0 with its automatic ontology) runs on a seeded sample of held-out products because its graph build costs about 0.55 USD per product: 6 products: AutoTuneForce, CoachForce, EdgeForce, PitchForce, ProposalForce, WorkFlowGenie. Under the rule in section 11 only one question family is multi-hop, so H3 rests on few questions (section 11). A4 answers every held-out question of the multi-hop families in all products, and otherwise runs, with A4c and A0f, on one seeded prefix of the held-out products (seed 20261015), sized from the pilot costs so the whole project stays under the cap; the table above lists the products. Any arm run on a product subset is compared with A5 and A0 on the same products only, and its denominators are those products' questions.

How the differences are read:
- A0f minus A0: the effect of the pre-extracted representation on a retrieval agent (representation and retrieved content change together).
- A4 minus A0f: the same cards, typed access and exact aggregation against retrieval.
- A5 minus A4: the full multi-hop toolkit (multi-hop queries, aggregation across cards, algorithms) against the restricted card view; not attributed to multi-hop alone.
- A5 against A4c: multi-hop inside fixed tools against the model assembling it step by step (accuracy, calls, tokens, latency).
- Every difference is measured under the registered resource limits. An arm that stops on its budget more often is not given more; budget-stop rates are reported per arm.
- None of these differences shows that another database could not do the same. A3 (SQLite over the same cards and tool semantics) checks that row for row.

## 8. Endpoints and hypotheses (24 held-out products)

- Primary endpoint: the macro average of person, company and artifact (PR and URL) questions; each type is the mean of F1 x 100 over its questions, then the three types are averaged with equal weight.
- Paper-style endpoint, reported alongside: the four-type macro including content questions. Unanswerable questions are reported separately.
- H1: A5 minus A0 on the primary endpoint, 95% interval lower bound above 0. Prior expectation: at least 15 points.
- H2: A5 minus A1f, 95% interval lower bound above -3 (non-inferiority), and A5's median input tokens per answerable question at most 10% of A1f's. Both parts must hold.
- H3: on the multi-hop families (section 11), A5 minus A4, 95% interval lower bound above 0. Prior expectation: at least 10 points. The difference over all questions is reported next to it.
- H4: A5's correct refusal rate on unanswerable questions (HERB's judge) is at least A1f's. Reported next to it for every arm: the explicit refusal rate (the answer's outcome is a refusal, decided without a model, because the judge can score an unrelated answer as a refusal) and the false refusal rate on answerable questions.
- A5 against A4c, A0f against A0 and A0v against A0 have no pre-set direction and are reported as they come.

## 9. Scoring

- Person, company, PR and URL questions: the structured answer_ids field is compared with the gold set after HERB's own `normalize`, and scored with HERB's set F1. No model extracts ids from prose.
- Content questions and unanswerable questions: HERB's judge prompts and score mappings, with model M as the judge (one judge for every arm). Scores are therefore not comparable with the HERB paper's GPT-4o-judged numbers; the paper's numbers are shown only for reference.
- Fixed denominators: all 659 answerable and all 544 unanswerable questions (or an arm's pre-registered product subset). A failed, timed-out or empty answer scores 0 on every type, including unanswerable questions, and failures are counted per arm.
- Judge errors are retried on a later grading pass and never become an arm failure.

## 10. Statistics

- Product-clustered paired bootstrap over the 24 held-out products, 10,000 replicates, seed 20261015. A replicate resamples products with replacement and keeps each drawn product's questions; both arms are recomputed on the same draw, paired by question key. Intervals are the 2.5 and 97.5 percentiles.
- A type with no question in a replicate is left out of that replicate's endpoint, and such replicates are counted.
- Also reported: results by seen and unseen question strings and by type; overlap of cited records with HERB citations (descriptive only); input and output tokens, tool calls, p50 and p95 latency per question.

## 11. Multi-hop families (H3)

A family is a normalised question string. A family is multi-hop when the answer values its questions need are not in what A4 can see: the cards of every root the question text can name (plus the Product), and the full cards of their one-hop neighbours. On the development set this was computed by code for every instance; a family with mixed instances counts as multi-hop only if every instance needs content outside the visible set. Held-out strings never seen in development were classified from the question text and the card field list only, independently by two models (Codex and a Claude subagent); only strings both called multi-hop are in the set, and disagreements or uncertain cases stay outside it (they remain in every other endpoint). This is a schema-only classification, not a check against held-out content.

A first Claude classification attempt was discarded unused because the agent assigned verdicts by keyword rules instead of reading each template; a fresh agent then classified every template. Both final classifications agree on all 215 unseen strings.

Under the unchanged visibility rule, the development set yielded 1 multi-hop, 71 single-hop and 26 content families; 94 of 99 voting instances were visible. A paging diagnostic found 90 of 332 answer values visible only inside neighbour groups larger than one page (200 cards or 12,000 tokens), and 15 families covering 31 development questions with every instance in that state or missing. This does not establish one-call inaccessibility, because A4 can filter, count and group before paging, so it defines no subgroup.

332 question families classified (content (not classified) 45, multi 1, single 286); multi-hop set: 1.

multi-hop family: find employee ids of team members who provided insights on the strengths and weaknesses of <p>'s competitor products

Held-out strings excluded after disagreement or uncertainty:

none

## 12. Development acceptance

Extraction was accepted on the six development products by two models, not by people: auditor 1 Codex, auditor 2 a Claude subagent, each reading only the audit packet. An item is correct when both say correct; a disagreement goes to a third Codex pass that sees both reasoned verdicts and decides. The gates use the adjudicated result. Agreement between two models does not show equivalence to human labels, and the third pass shares a model family with auditor 1.

The acceptance took six rounds. Each round drew a fresh seeded sample (seed 20261015 plus the round number), and every failed round is kept with its packets and verdicts:

| Round | Rubric | Failed gates | Passed gates |
|---|---|---|---|
| 1 | v1 | bug events 42/50, review relations 36/50, bug recall 104/146, reviewer recall 8/15 | names 50/50, newly unowned 30/30, dependency edges 27/30, affected paths 2/2 |
| 2 | v2 | review relations 41/50, dependency edges 25/30, affected paths 0/2, reviewer recall 0/0 (no reviewers in the sampled documents) | bug events 49/50, names 50/50, newly unowned 30/30, bug recall 111/115 |
| 3 | v3 | reviewer recall 8/24 | every precision gate, bug recall 119/124 |
| 4 | v3 | reviewer recall 24/52 | every precision gate, bug recall 47/48 |
| 5 | v4 | review relations 39/50 | every other precision gate, bug recall 142/146, reviewer recall 35/39 |
| 6 | v5 | none | all (table below) |

Between rounds, extraction changed only by deterministic rules; no prompt changed. Re-deriving the facts reused the cached model outputs, except about 0.13 USD of feature-membership calls whose inputs depend on earlier outputs, and one accidental call of 0.0005 USD during development. The rules: one ASSIGNED event per assignment, on the message that hands the bug over; component evidence from the acceptance; review relations limited to content proposals and review questions about this product's documents (excluding work allocation, competitors' documents, author acknowledgements, summaries and openers), unversioned remarks attached to every version of the type; a rule that reads type-level discussions the model left unread (its edges are marked review_origin="type_level_rule"); name candidates taken from the meeting's attendees first, as the design states; explicit task statements outranking title overlap for PR membership; completion statements read for feature fate. The rubrics changed only to state definitions the auditors disagreed on, or to add evidence the packets lacked: the bug event model, the frozen state predicates and candidate sets for ambiguous names (v2); product provenance records, development membership and recall eligibility (v3); author acts and type-level reviewer recall (v4); product identity including internal names, and the name rule as implemented (v5). After round 4 the reviewer-recall endpoint was amended: targets are document types (all versions), author acknowledgements are excluded by act, and strict per-document recall and distinct-reviewer coverage are reported alongside; rounds 1 to 4 keep their original, failed results. After round 2 the only two affected paths were found to rest on a feature the records report as completed; with that fixed, the dev pool holds no affected path, so that gate is reported as not estimable. Malformed auditor answers were retried once each (3 packets in round 3, 1 in round 4, 2 in round 5).

Audit round 6; sample hash 34251f49d0d932fd6ba18e6f3a54e27b85283899a676f25d7f38f7c73d98f359; summary sha256 ce4c3d7e76604929b3f20594d9b2166496888dd5fa4e0aec836473b8555e9aa6.

Auditors (exact model ids and settings):
- claude: {"model": "claude-sonnet-5-5", "prompt_hash": "5265542fbc58c8e776f039daea52b4e4311242568e9cc1272007895a82c2bea7", "settings": "one fresh agent per batch, two at a time; reads only the batch file and writes only its verdict file; no other tools or files", "version": "Claude Code workflow agent (model alias sonnet)"}
- codex: {"model": "gpt-6.1-sol", "prompt_hash": "fce3310b100d74dd845bf22ee06cff4e67540adb2877ee2b4662063ba28d422b", "settings": "codex exec -s read-only, model_reasoning_effort=high, one fresh process per batch started in an empty directory holding only the batch file; one retry when the reply is not JSON", "version": "codex-cli 0.162.1"}
- third: {"model": "gpt-6.1-sol", "prompt_hash": "fce3310b100d74dd845bf22ee06cff4e67540adb2877ee2b4662063ba28d422b", "settings": "codex exec -s read-only, model_reasoning_effort=high, one fresh process per batch started in an empty directory holding only the batch file; one retry when the reply is not JSON; third pass over third_batches (shares the model family with auditor 1)", "version": "codex-cli 0.162.1"}

Prompt hashes:
- common_precision: 51da091166f516ba0cc9ad46dc4c742421a34e4f9950478b3c8b749db9f7808c
- common_recall: 0e3480c20d3c7bb9c172379cad668dad0513ff4771731956f064d1939b9ea926
- open_work: 449aacfb21be7e39bdb3df08fcfaf8a5a015fdede550d84a0b9e05809d528359
- recall_rules:recall_bug: 673756c2fa5f50cabda32a9e6cc21085f0c4a7a43a58c1abd21ddfbe0415896f
- recall_rules:recall_reviewer: cbe6294c4d1aaeb2b2d8049d1358a320ffd45e919ea72faeade9e6b47469c5bb
- rubric:bug_event: 371137e11726258ab47bcf9e143b5c71883ecf806fdf0df404229a004d0700ba
- rubric:name_resolution: 02abe5c837f2fab89cda5402407964d0177f24987b09429a60cc8a026c261c09
- rubric:review_relation: 2c5bb5ee41266f8b48721f7f5a651216ea0f76b431d773ffbe791a8720eb1e6b
- rubric:whatif_edge: ac12e5062905da7a6717e7a2b71b2ec98edafbf857b7282f61b8ce927ee7d29b
- rubric:whatif_path: 82ab78116e7ec3c122e7f49fd02de45467dad9694e5838aeccca6c15c118703f
- rubric:whatif_unowned: e7e3e2621b2a77cb2979006765349d48cfbc756f3d342883b858c9c9ad912f45
- third_precision: df0b74dafd6300b8793d02debb0fa81fcce869336097f051ff3dbb0c8fdaf6e2
- third_recall: 31c858f27f902632a6750b5a1dc103c09bd3d5be97430f64ea3958ded14c1bdc
- version: audit-prompts/v5

| Gate | Numerator | Denominator | Threshold | Result |
|---|---|---|---|---|
| audit:precision:bug_event | 50 | 50 | 0.9 | pass |
| audit:precision:review_relation | 50 | 50 | 0.9 | pass |
| audit:precision:name_resolution | 50 | 50 | 0.9 | pass |
| audit:precision:whatif_unowned | 30 | 30 | 0.9 | pass |
| audit:precision:whatif_edge | 29 | 30 | 0.9 | pass |
| audit:precision:whatif_path | 0 | 0 | 0.9 | not estimable (empty dev pool) |
| audit:recall:recall_bug | 75 | 76 | 0.85 | pass |
| audit:recall:recall_reviewer | 42 | 48 | 0.85 | pass |
| audit:complete | 1 | 1 | 1 | pass |
| dev score: A5 person and company macro | 86.16 | 100 | 75.0 | pass |
| freeze check | 0 | n/a | 0 differences | pass |
| what-if verification mismatches | 0 | 60 | 0 mismatches | pass |
| families.json present | 1 | 1 | 1 | pass |

## 13. Deviations from the design, with reasons

1. FalkorDB 4.22.1 in Docker stands in for FalkorDB Cloud 4.22; 6.0.2 is the primary server. Both versions are measured.
2. Read-only access uses ACL rules on graph-name patterns. FalkorDB has no direction-restricted GRAPH.COPY rule, so the what-if user's copy rights are pinned by tests instead.
3. Human acceptance labels (two people, 30 overlapping items) are replaced by the two-model audit in section 12. The held-out family labels by two people are replaced by the two-model text classification in section 11.
4. The human check of 50 grader extractions per arm is not done: id types are scored from the structured answer_ids field, so no grader extraction exists to check. The judge is model M, not GPT-4o.
5. A failed or empty answer scores 0 on unanswerable questions too. HERB counts an empty answer as a correct refusal.
6. There is no Review node: a review has no identity beyond its PR and position, so REVIEWED is a Person to PR edge with a per-review discriminator. An IN_CHANNEL edge (Message to Channel) is added.
7. A0 runs the HERB paper's retriever and 8 metadata tools inside our shared agent loop, with the same limits as our arms; every model call in it uses model M.
8. The what-if feature excludes documents from open work items: feedback on a document does not show that it is still unresolved. This feature is not scored.
9. The answer guard validates ids and structured answer fields; it does not check names written in prose.
10. Budget: Nebius bills the model; the project is capped at 50 USD, so arm sub-caps and any product subsets are fixed in section 7 from a recorded dev pilot.
11. Name resolution scores an @-mention in the meeting, not being answered by name (the design also counted the latter); candidates are attendees first, as the design states.

## 14. What we will say for each outcome (written now)

- H1 and H2 hold: "On Salesforce's HERB questions, X points above an agent using the same model, no worse than 3 points below reading the whole product, with Y% of its tokens, and a path behind every answer."
- H1 holds, H2 does not: "X points above an agent using the same model. Against reading the whole product the difference is D, 95% interval [L, U]; non-inferiority is not established. We use Y% of its tokens."
- H3 does not hold: "No traversal gain detected on the multi-hop families (difference D, interval [L, U])." The write-up then centres on extraction, name resolution and evidence chains, as measured.
- H3 not estimable (no multi-hop family): "Under the pre-registered rule no question family needs content beyond the depth-1 card view, so H3 is not estimable. The all-questions A5 minus A4 difference is D, interval [L, U]."
- H1 does not hold: we do not lead with scores; the post publishes the negative result.

## 15. After this file

The held-out split is run once per arm, in the order A5, A4, A0, A1f, A0f, A4c, A0v, A2, through one preflight that checks this file's hash, the release record, the acceptance report, a clean tree and the freeze manifest. Every run writes receipts carrying the hash of the extracted facts given in section 2; `make audit` checks them.
