# Pre-registration: inherit-brain on HERB

Generated {{generated_utc}} from commit {{code_commit}}. Nothing below changes after this file's hash is posted. Any later edit is published as a separate, dated "post-hoc version" and never replaces this one.

## 1. What this file fixes

This file fixes, before any held-out question is answered by any arm: the inputs and their hashes, the extracted facts, the code and prompts, the arms and their budgets, the endpoints and hypotheses with their thresholds, the scoring and statistics, the multi-hop family list, and the sentence we will publish for each outcome.

Time evidence: this file is committed to a private repository; its sha256 is published in a short public post, and the post is archived in the Wayback Machine. The held-out run starts only after both exist and are recorded in `results/freeze/release.json`; the run refuses to start otherwise.

## 2. Identity

| Item | Value |
|---|---|
| Code commit | {{code_commit}} |
| Upstream HERB commit | {{upstream_herb_commit}} |
| Split hash | {{split_hash}} |
| Input bundle sha256 | {{bundle_sha}} |
| Freeze manifest sha256 | {{manifest_sha}} |
| Effective configuration hash | {{config_hash}} |
| Acceptance report sha256 | {{acceptance_sha}} |

facts hash: {{facts_hash}}

Versions of every prompt and protocol:

{{versions_table}}

## 3. Data and split

- HERB (Salesforce AI Research), 30 products. Six products are the development set: ActionGenie, AnomalyForce, ContentForce, KnowledgeForce, LeadForce, TrendForce. The other 24 are held out.
- Held-out questions: 659 answerable and 544 unanswerable. Of the answerable ones, 416 have a normalised question string that also occurs in the development set and 243 do not.
- Inputs allowed into extraction and the graph: each product's slack, documents, meeting_transcripts, meeting_chats, urls and prs, plus metadata/employee.json and metadata/customers_data.json. Excluded: the products' team and customers fields, the questions, ground truth and citations. The bundle is built from the allowed fields only; its hash is above.
- All 30 products were extracted and loaded with the same frozen code before this file was written. No held-out question was read by any tool, arm or person during development.

## 4. Model and limits

- Every arm uses model M, `Qwen/Qwen3-30B-A3B-Instruct-2507` on Nebius, at the settings in the configuration hash. Embeddings: `Qwen3-Embedding-8B`, 1,024 dimensions.
- Every model call goes through one local metering proxy with a 50 USD hard cap for the whole project and a sub-cap per arm (section 7). A call that would cross a cap is refused.
- Agent arms (A5, A4, A4c, A0, A0f) share one loop with the same limits: at most 12 tool calls, 24 model calls, 60,000 cumulative input tokens and 300 seconds per question; 1,024 output tokens per model call. The input budget was raised from 40,000 after a dev pilot in which A4 stopped on it in 22 of 48 questions (pre-registered rule: above a quarter, raise it for every loop arm and pilot again). In the second pilot, at 60,000, A4 stopped in 41 of 48 (it uses whatever budget it gets), A0 in 17 of 48 instead of 33, and no loop arm's pilot score rose. The budget stays at 60,000 because it starves A0, the comparator of H1, less; raising it further would not fit the 50 USD cap.
- A1f, A0v and A2 answer with one model call over their own context. A1f's call holds the whole product (about 110,000 to 140,000 tokens), so the loop's cumulative budget does not bind it; its limit is the model's 262,144-token context.

## 5. The four frozen protocols

1. Bug states. Events are ordered by message time; types are REPORTED, ASSIGNED, RESOLVED (with a fix PR), DISMISSED and REOPENED. "Current" means the last day of the corpus. Unresolved: no linked fix PR. Active: the last event is REPORTED, ASSIGNED or REOPENED. Reopened: at least one REOPENED event. No fix needed: at least one DISMISSED event.
2. Time. Every event takes its message timestamp; the UDF `herb_v1.statusAt(events, date)` gives a bug's state on a date under the rules above.
3. Name resolution. Candidates are co-attendees of the meeting or posters in the same channel. Score = 3 x role cue (the spoken role matches the directory role) + 2 x the candidate posted in a related Slack thread within 3 days + 1 x the candidate was mentioned or answered by name in the same meeting. A unique top score resolves the name; a tie keeps every tied candidate, marked ambiguous, and the answer says so.
4. Refusal. An arm answers "not in the records" only when an entity the question names cannot be resolved, or when every relation the question needs returns nothing. Partial results are answered as partial. Every arm gives the same refusal sentence, "I could not find information in the records to answer this question.", with empty answer_ids.

## 6. Entity cards

A card is a node's own fields plus, for each directly connected node, its type, id and name (not its fields), and each edge's source record ids. The A0f retrieval corpus and the A4 tools read the same cards. The complete field list:

{{card_fields}}

Choices that bound what a card shows: Message neighbours are omitted from every non-Message card (messages are evidence, not entities); the derived COLLABORATED edge is excluded; at most 3 source record ids are listed per neighbour entry, then "..."; the Product card lists only bugs, documents, meetings, features, PRs and competitors, with no messages and no counts; one depth-1 response returns at most 200 rows and 12,000 tokens, with a cursor for the next page.

## 7. Arms

Every arm answers in the same JSON shape with the same answer instructions (prompt versions above): answer_text, answer_ids, outcome, then at most 10 distinct evidence ids. Only the answer fields are scored. A reply cut off inside the final evidence list is accepted when every graded field before it is complete; graded fields are never repaired.

{{arms_table}}

A2 (GraphRAG-SDK 1.4.0 with its automatic ontology) runs on a seeded sample of held-out products because its graph build costs about 0.55 USD per product: {{a2_sample}}. No question family is multi-hop under the rule in section 11, so H3 is not estimable and A4 carries no confirmatory hypothesis. A4, A4c and A0f therefore run on one seeded prefix of the held-out products (seed 20261015), sized from the pilot costs so the whole project stays under the cap; the products are listed in the table above. Any arm run on a product subset is compared with A5 and A0 on the same products only, and its denominators are those products' questions.

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

{{dev_visibility_summary}}

{{family_counts}}

{{family_block}}

Held-out strings excluded after disagreement or uncertainty:

{{excluded_unseen}}

## 12. Development acceptance

Extraction was accepted on the six development products by two models, not by people: auditor 1 Codex, auditor 2 a Claude subagent, each reading only the audit packet. An item is correct when both say correct; a disagreement goes to a third Codex pass that sees both reasoned verdicts and decides. The gates use the adjudicated result. Agreement between two models does not show equivalence to human labels, and the third pass shares a model family with auditor 1.

{{auditors}}

{{acceptance_table}}

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

## 14. What we will say for each outcome (written now)

- H1 and H2 hold: "On Salesforce's HERB questions, X points above an agent using the same model, no worse than 3 points below reading the whole product, with Y% of its tokens, and a path behind every answer."
- H1 holds, H2 does not: "X points above an agent using the same model. Against reading the whole product the difference is D, 95% interval [L, U]; non-inferiority is not established. We use Y% of its tokens."
- H3 does not hold: "No traversal gain detected on the multi-hop families (difference D, interval [L, U])." The write-up then centres on extraction, name resolution and evidence chains, as measured.
- H3 not estimable (no multi-hop family): "Under the pre-registered rule no question family needs content beyond the depth-1 card view, so H3 is not estimable. The all-questions A5 minus A4 difference is D, interval [L, U]."
- H1 does not hold: we do not lead with scores; the post publishes the negative result.

## 15. After this file

The held-out split is run once per arm, in the order A5, A4, A0, A1f, A0f, A4c, A0v, A2, through one preflight that checks this file's hash, the release record, the acceptance report, a clean tree and the freeze manifest. Every run writes receipts carrying the hash of the extracted facts given in section 2; `make audit` checks them.
