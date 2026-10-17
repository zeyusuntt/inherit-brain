# Results: inherit-brain on HERB (held-out run)

This run follows the pre-registration in [PREREG.md](../PREREG.md), sha256 `d8f4363276e972bbfb9bc2714fcceae128ddede4d163b00ffe4aaac602eadffd`, committed at `bb5dcce` and posted before any held-out question was answered: [public post](https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67), [Wayback snapshot](https://web.archive.org/web/20261017025021/https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67). Every arm ran once, in the pre-registered order, through the launch preflight. Per-slot tables are in `results/tables/`, run receipts in `results/receipts/`, and every number below is in `results/results.json`.

## Outcome (the sentence written in advance)

H1 and H2 hold:

> On Salesforce's HERB questions, 40.8 points above an agent using the same model, no worse than 3 points below reading the whole product, with 8.0% of its tokens, and a path behind every answer.

Against reading the whole product (A1f) the difference is in fact +54.0 points, not a non-inferiority margin. That gap is partly a format failure of A1f (see the second post-hoc analysis): with its company answers mapped from customer ids to names it is still +42.7.

## Hypotheses (24 held-out products, product-clustered paired bootstrap, 10,000 replicates)

| | Comparison | Difference | 95% interval | Threshold | Result |
|---|---|---|---|---|---|
| H1 | A5 minus A0, primary endpoint | +40.8 | [35.7, 46.2] | lower bound > 0 (prior: at least 15) | holds |
| H2 | A5 minus A1f, primary endpoint | +54.0 | [48.7, 58.8] | lower bound > -3 | holds |
| H2 | A5 median input tokens / A1f's | 8.0% (10,117 / 125,947) | | at most 10% | holds |
| H3 | A5 minus A4, multi-hop family | +90.2 | [84.5, 95.8] | lower bound > 0 (prior: at least 10) | holds, on 7 questions |
| H3, all questions | A5 minus A4 on every question A4 answered | +35.6 | [27.2, 47.9] | reported, no threshold | |
| H4 | correct refusal rate (HERB judge) | A5 82.4%, A1f 13.8% | | A5 at least A1f | holds |

H3 rests on one multi-hop family with 7 held-out questions (all person questions, one per product); its interval is narrow only because A5 scored between 80 and 100 on all seven (three fully correct) while A4 scored 0 on six and 10.5 on one, so read it as a small, pre-registered check, not a general finding about traversal.

Paper-style endpoint (four types, with content questions): H1 +36.1 [32.1, 40.2], H2 +44.3 [40.3, 47.9].

## Per arm

| Arm | Questions | Primary | Person | Company | PR | URL | Content | Unanswerable (judge) | Explicit refusal | False refusal | Failed | Median input tokens | Budget stops | Latency p50 / p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5 | 1,203 | 80.0 | 82.5 | 87.8 | 66.4 | 100.0 | 74.1 | 82.4 | 80.5 | 7.0 | 0 | 10,117 | 1.3% | 4.7 / 16.1 |
| A0 | 1,203 | 39.2 | 34.6 | 47.7 | 35.9 | 28.7 | 52.1 | 37.7 | 35.9 | 9.7 | 44 | 46,451 | 44.5% | 21.4 / 71.2 |
| A1f | 1,203 | 26.0 | 27.0 | 10.0 | 40.4 | 46.7 | 58.8 | 13.8 | 9.9 | 3.0 | 49 | 125,947 | n/a | 11.6 / 50.6 |
| A0v | 1,203 | 19.2 | 28.4 | 0.0 | 26.7 | 54.4 | 45.0 | 65.4 | 62.0 | 28.7 | 9 | 5,176 | n/a | 5.1 / 18.2 |
| A4 | 280 | 27.3 | 20.8 | 40.7 | 14.9 | 100.0 | 28.4 | 71.7 | 71.1 | 43.0 | 0 | 62,033 | 68.9% | 11.3 / 31.9 |
| A4c | 275 | 28.2 | 28.4 | 38.9 | 14.1 | 66.1 | 26.7 | 69.2 | 65.4 | 44.8 | 0 | 60,976 | 67.6% | 14.3 / 37.7 |
| A0f | 275 | 42.3 | 54.1 | 29.6 | 40.3 | 87.5 | 65.7 | 21.4 | 17.0 | 1.7 | 4 | 64,164 | 79.6% | 20.8 / 43.9 |
| A2 | 254 | 24.0 | 29.8 | 8.9 | 33.2 | 33.3 | 51.7 | 59.6 | 58.5 | 13.1 | 8 | 11,142 | n/a | 7.0 / 36.5 |

Scores are F1 x 100 (ids) or the HERB judge on model M (content, unanswerable). "Failed" counts run failures, scored 0 and kept in the denominator. A4, A4c and A0f ran on five seeded held-out products (CollaborateForce, ConnectForce, FeedbackForce, MonitorForce, SecurityForce), A4 also on the 7 multi-hop questions; A2 on its six sampled products. Their columns are not comparable to the full arms; the contrasts below pair them with A5 or A0 on the same questions.

## Secondary contrasts (same questions, primary endpoint)

| Contrast | Difference | 95% interval |
|---|---|---|
| A0f minus A0 (five products) | +20.7 | [14.9, 26.4] |
| A4 minus A0f (five products) | -13.8 | [-26.6, -0.5] |
| A5 minus A4c (five products) | +33.5 | [21.5, 46.1] |
| A0v minus A0 (all products) | -19.9 | [-23.2, -15.9] |
| A5 minus A2 (A2's six products) | +58.8 | [45.3, 69.8] |
| A0 minus A2 (A2's six products) | +16.4 | [2.5, 24.7] |

How to read them (as registered): A0f minus A0 is the effect of the pre-extracted card representation on the same retrieval agent. A4 minus A0f compares typed depth-1 access with retrieval over the same cards; here typed access did worse, and A4 hit its input budget on 68.9% of questions. A5 minus A4 and A5 against A4c measure the full multi-hop toolkit against the card view, not multi-hop alone. None of these shows that another database could not do the same.

## Also reported (pre-registered in PREREG section 10, no threshold)

These two items were registered as "also reported" and were missing from the first version of this page. They come from the same graded runs, with no model call (`results/held_run/reported_extras.py`, output in `results/reported_extras.json`).

Seen and unseen question strings. 610 held-out questions share their normalised question string with a development question; the tools were built against development questions, so this is where overfitting would show.

| Comparison | Seen in development (610 questions) | Never seen (593 questions) |
|---|---|---|
| H1, A5 minus A0 | +57.9 [51.8, 63.7] | +20.1 [12.9, 26.5] |
| H2, A5 minus A1f | +58.7 [50.0, 66.6] | +44.1 [40.4, 47.7] |
| A5 primary endpoint | 88.9 | 79.2 |
| A0 primary endpoint | 31.0 | 59.2 |

The lead over A0 is about a third as large on strings never seen in development, and still above zero. The two halves are not the same mix of questions: 135 of the 243 answerable unseen questions are PR questions, against 10 of 416 seen ones, so the split mixes novelty with question type.

Cited records against HERB's citations (answerable questions, descriptive only, never graded). Precision is the share of an answer's cited records that HERB also cites; recall is the share of HERB's citations the answer cites. HERB lists many records per question, so recall is low for every arm.

| Arm | Questions | Precision | Recall | No evidence cited |
|---|---|---|---|---|
| A5 | 659 | 74.8% | 7.8% | 46 |
| A0 | 642 | 35.9% | 3.5% | 61 |
| A1f | 627 | 22.3% | 4.1% | 109 |
| A0v | 652 | 35.7% | 3.4% | 185 |
| A2 (six products) | 152 | 31.2% | 3.3% | 30 |

A4, A4c and A0f cite cards, not records, so their overlap (2% to 10% precision) says little.

## Post-hoc analyses (not pre-registered)

1. Provider network errors. 44 of A0's and 6 of A1f's failures were the proxy's "502 upstream unreachable" errors from the model provider, not the arms' own failures. They stay zero-scored above, as registered. Dropping those slots from both arms of the pair: H1 +40.3 [35.0, 45.9], H2 +54.3 [49.0, 59.0].
2. Company answers given as customer ids. All arms were told to answer a company by its name, never its CUST- id; A1f gave CUST- ids in 49 of its 104 company answers (A0 in 3, A5 in none), which score 0 under the registered id matching. Mapping CUST- ids to company names through the customer directory, the same way for every arm, and rescoring with the same F1: primary endpoint A5 80.0, A0 39.6, A1f 37.3; H1 +40.4 [35.5, 45.6], H2 +42.7 [36.9, 48.1]. The conclusions do not change.

## What to keep in mind

- Development acceptance was model-based (two model auditors and a third pass), took six rounds, and amended the reviewer-recall endpoint after round 4; PREREG section 12 lists every round.
- A1f failed on 49 questions: 43 replies were invalid JSON that the shared salvage rule does not repair (a repetition loop cut inside the answer text or before the first id), 6 were provider errors. Long-context answering with this 30B model is fragile in exactly this way.
- The loop arms share a 60,000-token input budget. A0 stopped on it in 44.5% of questions, A5 in 1.3%. The differences are measured under that registered limit; an arm that runs out of budget is not given more.
- The judge is model M, so content and unanswerable scores are not comparable with the HERB paper's GPT-4o-judged numbers.
- Total model spend for the whole project, through grading: 42.72 USD of the 50 USD cap.
