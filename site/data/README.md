# Demo data

Static JSON for the demo site, written by `uv run python site/build_data.py` from the finished held-out run
(results/, runs/eval/<ARM>-held, the bundle in data/bundle, and the 6.0 FalkorDB graphs, read only). No model calls.
Everything is derived from HERB data (Salesforce AI Research); the raw dataset is not included.

Attribution: HERB (Salesforce AI Research), CC BY-NC 4.0.

| File | Contents |
|---|---|
| `overview.json` | Per-arm summary (primary endpoint, per-type means, unanswerable, explicit and false refusal, failures, median input tokens, budget-stop rate, latency p50 / p95, n), H1-H4 table, secondary contrasts, both post-hoc analyses, the PREREG sha256, gist and Wayback links, total spend (42.72 USD), arm descriptions. |
| `questions.json` | One row per held-out question (1,203): `k` key (`<product>/<ans\|unans>/<index>`), `p` product, `t` type, `q` text, `seen` flag, `s` score per arm (null where the arm did not answer it). |
| `q/<product>.json` | One file per held-out product (24). `questions`: gold (ids and names for id types, reference text for content, null for unanswerable), each answering arm's outcome, answer text, answer ids and names, score and failed flag, and for A5 the tool path (each call in order: tool, parameters, rows, milliseconds, the Cypher queries it ran) plus diagnostics (model calls, tool calls, input tokens). `evidence`: record excerpts (id, kind, title, author, timestamp, text cut to 600 characters) for A5's evidence ids and the record ids in its tool rows, from the bundle. |
| `whatif.json` | "If this person leaves", for up to 3 Engineering Leads per product: newly unowned open items, affected items with the dependency path, with titles. Each product also has `owners`: the 5 people (any role) whose departure leaves the most open items unowned, in the same shape as `leads`; `owners_note` says how they were picked. |

Notes

- Scores are 0 to 100: F1 for id types, the HERB judge on model M for content and unanswerable. A failed slot is 0.
- Tool-to-query assignment in `a5.path` is exact except in the few slots flagged `ops_assignment_ambiguous`, where
  two consecutive calls of the same tool could not be told apart in the run's flat query log.
- What-if groups come from a read-only export of the graph and the independent Python check
  (`inherit.whatif_check`), which matched the graph counterfactual on every lead in results/freeze/whatif_check.json.
  They are computed from recorded responsibility and dependency, not a prediction of real consequences.
- Text fields are truncated: record excerpts 600 characters, answers 1,500, content reference text 2,000.
