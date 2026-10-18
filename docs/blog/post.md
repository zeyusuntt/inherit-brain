---
title: "What retrieval missed: a company brain on FalkorDB, tested on 1,203 held-out questions"
published: false
description: "I built a company brain with fixed Cypher tools, tested it against retrieval on HERB, and learned where joins and extraction rules mattered."
tags: falkordb, wemakedevs, graphhacks, ai
cover_image: https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/cover.png
---

I asked my agent for the authors and key reviewers of a product's technical spec. It returned all 10 employee IDs. Retrieval with the same model returned one. Two of those people were named Alice Davis.

I built inherit-brain for Graph Hacks, Track 3: Company Brain. I used Salesforce's HERB benchmark, a simulated company with 30 products. I developed on six products and tested on the other 24. I wanted to see what the graph tools recovered, what retrieval missed, and where both failed.

Recorded demo, including answers and tool traces: https://zeyusuntt.github.io/inherit-brain/
Code: https://github.com/zeyusuntt/inherit-brain

## What it does

inherit-brain turns a product's records into one FalkorDB graph per product: people, documents, meetings, Slack messages, pull requests, bugs and their event histories, features, customers, and who did what to which. An agent answers questions by calling nine read-only tools. Each tool runs fixed, parameterised Cypher; the model never writes a query.

I did not load the records directly as facts. Plain code extracts structure, resolves names and applies rules. Model calls extract bug histories and semantic relations such as review suggestions. I load those facts into FalkorDB and keep source record IDs on the edges.

For the spec question on one product, SupportForce, the agent made one tool call (`doc_people`), which ran three read-only queries in 21.9 ms. It returned ten people, each with the record that puts them there. The two Alice Davises have different employee ids; the graph kept them apart because a person node is an employee id, not a name.

![The overview of the demo site](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/cover.png)

## How FalkorDB fits in

I loaded 30 product graphs named `herb:<product>`, containing 56,708 nodes and 164,698 edges across 16 node labels and 24 edge types. Each edge stores `src`, a list of source record IDs that tools can return with their results. In the scored run, 46 of the 659 answers to answerable questions cited no record.

The agent connects as a read-only user per product, `agent_<product>`, created with

```
ACL SETUSER agent_<p> reset on ><pw> (~herb:<p> +GRAPH.RO_QUERY)
```

so it can run `GRAPH.RO_QUERY` on its own product's graph and nothing else. A question about one product cannot read another.

One held-out question asked who wrote the most fixes for bugs that customers later reopened in SecurityForce. The agent first tried `pr_board` with an invalid customer-reopen argument. That call was rejected. A second `pr_board` call returned no rows. Its third call used `bug_board`, which ran this join:

```cypher
MATCH (b:Bug) WHERE b.reopened = true
  AND ($components IS NULL OR b.component IN $components)
  AND ($companies IS NULL OR any(x IN b.companies WHERE x IN $companies))
MATCH (pr:PR)-[fx:FIXES]->(b)
MATCH (p:Person)-[op:OPENED]->(pr)
MATCH (b)-[hs:HAS_EVENT]->(rs:BugEvent) WHERE rs.type = 'RESOLVED' AND pr.link IN rs.fix_pr_links
MATCH (b)-[:HAS_EVENT]->(ro:BugEvent) WHERE ro.type = 'REOPENED' AND ro.ts > rs.ts
RETURN p.eid, p.name, p.role, b.id, pr.link, op.src, fx.src + hs.src, [rs.id]
```

Cypher joins the bug, its fixing PR, the author, and a resolution followed by a reopening. Python then counts distinct PR links per author and keeps every tie at the maximum. Julia Miller had two fixing PRs across the four matched bugs, which gave HERB's gold answer. Retrieval named four other engineers. The complete answer took 9.43 seconds across four model calls; the main join took 13.0 ms in this recording.

`resolve_entity` uses full-text search (`db.idx.fulltext.queryNodes`) as a fallback for bugs, features, documents and competitors. Person resolution matches employee IDs and normalised names, with meeting evidence for speakers. The `herb_v1.statusAt(events, $as_of)` UDF computes status from recorded events at a cutoff date. My `bridges` tool ranks directed betweenness on collaboration edges; its ranking depends on their orientation, and no held-out question called it.

For "if this person leaves", a separate what-if user may use GRAPH.COPY to copy its product's graph into a scratch namespace, delete a person in the copy, and diff who owns the open bugs and PRs before and after. On SupportForce, if Julia Smith leaves, two open bugs and one open pull request have no owner left. My independent Python check reported no mismatches. This feature computes consequences from recorded ownership and dependencies; it was outside the scored question run.

![SupportForce: if Julia Smith leaves, two open bugs and one open PR lose their only recorded owner](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/whatif.png)

Everything runs on FalkorDB 4.22.1 and 6.0.2; the tool tests run against both.

## Testing it so I could not fool myself

A demo is easy to cherry-pick, so I tested on a benchmark I did not write: Salesforce AI Research's [HERB](https://github.com/SalesforceAIResearch/HERB), a simulated company with 30 products, their records, and questions with gold answers (including questions that have no answer in the records). I built on six products and held back the other 24, with 1,203 questions.

Before the held-out run I froze the code, the prompts and the graphs (each hashed), wrote down the result I expected and what I would say for every outcome, and posted the plan's sha256 publicly (a [gist](https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67) and a [Wayback snapshot](https://web.archive.org/web/20261017025021/https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67)). A preflight refused to start any run whose hashes had drifted.

I compared eight methods using Qwen3-30B-A3B-Instruct-2507 on Nebius. Four covered all 1,203 questions: inherit-brain, HERB's retrieval tools in the shared loop, whole-product prompting, and HERB's hybrid retrieval. Four tested other tool designs on smaller question sets fixed before the run: depth-1 cards, chained card access, retrieval over cards, and GraphRAG-SDK with its automatic ontology. The [full results](https://github.com/zeyusuntt/inherit-brain/blob/main/results/RESULTS.md) give each method's coverage. Metered Nebius calls, including extraction and grading, cost 42.72 USD against my 50 USD cap.

| Same model | Structured-answer score | Median input tokens per answerable question |
|---|---|---|
| inherit-brain | 80.0 | 10,117 |
| HERB's retrieval tools, same loop | 39.2 | 46,451 |
| Whole product in one prompt | 26.0 | 125,947 |

I scored 478 questions with structured answers using HERB's set F1, multiplied by 100. I averaged three groups equally: people, companies, and artifacts, which combines PR and URL questions. Token medians cover all 659 answerable questions.

With the same model, agent loop and limits, replacing my graph tools with HERB's retrieval tools reduced the score by 40.8 points, with a 95% interval of 35.7 to 46.2 (product-clustered bootstrap). This changes the tools and the extracted representation together. It does not compare database engines.

![The removal test on the demo site: same model, same loop, graph tools against retrieval tools](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/removal.png)

## What I learned

Giving the model graph access was not enough. On five held-out products, typed depth-1 access to the same entity cards scored 13.8 points below retrieval over those cards. Across its 280-question run, it hit the input budget on 68.9% of questions. A separate version let the model chain calls through the cards; my full tools led that version by 33.5 points on the same five products. I would package a complete join into a tool before asking the model to assemble it one step at a time. In the reopened-bug tool, Cypher performs the joins and Python counts distinct PR links. These comparisons measure whole toolkits, not traversal alone.

My tools fit the questions I saw. 610 of the held-out questions share their wording (with the product name swapped) with a development question. On those, inherit-brain led retrieval by 57.9 points. On the 593 never seen, by 20.1 (12.9 to 26.5). That is still a clear lead, about a third of the size, and the results page shows it next to the headline. The split also changes the question mix: 135 of the 243 answerable unseen questions are PR questions, compared with 10 of 416 seen ones. I cannot attribute the smaller lead to unfamiliar wording alone.

My extraction failed five audit rounds and passed the sixth. Two failures: an assignment needed an event on the message that handed the bug over, and an acknowledgement was not a review. Codex and a Claude agent judged sampled facts against raw records; a third Codex pass resolved disagreements. Between rounds I changed extraction rules and audit rubrics. After round four I also amended reviewer recall to target document types across versions and exclude author acknowledgements. Every round is summarised in [PREREG section 12](https://github.com/zeyusuntt/inherit-brain/blob/main/PREREG.md#12-development-acceptance). This was acceptance by model auditors on samples, not human validation.

A few FalkorDB details I did not expect:

- `%R~herb:p` ACL rules do not work for graph commands: `GRAPH.RO_QUERY` declares its key as read-write, so a read-only key rule never matches. A command-scoped selector, `(~herb:p +GRAPH.RO_QUERY)`, does.
- `GRAPH.COPY` ACLs cannot distinguish source from destination. My what-if user can copy between its product graph and scratch namespace. The product graph must remain present while that user is enabled: the existing destination prevents a copy back into its name. Tests check this boundary; they do not enforce that operational condition.
- The server's `TIMEOUT` was 1000 ms, so whole-graph identity reads (hashing a graph for the freeze) timed out until I passed a per-query timeout.
- On two small directed fixtures, 6.0.2 passed all five betweenness checks and 4.22.1 passed two. I use a Python fallback on 4.22.1.

I kept the [ACL reproduction and other integration feedback](https://github.com/zeyusuntt/inherit-brain/blob/main/docs/sponsor-feedback.md), the [version probes and betweenness checks](https://github.com/zeyusuntt/inherit-brain/blob/main/docs/falkor-versions.md), and the [freeze reader's timeout override](https://github.com/zeyusuntt/inherit-brain/blob/main/src/inherit/freeze.py#L349-L357) in the repository.

With this model, whole-product prompting had 49 failed answers: 43 invalid JSON replies and six provider errors. It also returned customer IDs instead of names in 49 of 104 company answers. In a post hoc check, I mapped customer IDs to names for every method; the whole-product score rose from 26.0 to 37.3. Check output-format failures separately before reading a low score as missing facts.

## Try it

The [explorer](https://zeyusuntt.github.io/inherit-brain/explore.html) contains all 1,203 held-out questions, HERB's gold answers, and the answer and score from each method that ran that question. It also shows inherit-brain's recorded tool calls and Cypher, including questions it got wrong. The site makes no model calls.

![One recorded SecurityForce question, its gold answer, and the methods' answers and scores](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/explorer.png)

With Git, Docker and uv installed, I can replay the recorded tool calls against local FalkorDB without a model key:

```bash
git clone https://github.com/zeyusuntt/inherit-brain.git
cd inherit-brain
uv sync
docker compose up -d
make quickstart
uv run python -m inherit.ask --replay runs/eval/A5-held --slot SecurityForce/ans/21
```

The same nine tools are available as an MCP server (`uv run python -m inherit.mcp_server --product SecurityForce`).

I built this with Claude Code and Codex as coding tools; the benchmark choice, experiment design, gates and release decisions were mine.

When you inherit a product, you inherit its history. inherit-brain lets you ask it.
