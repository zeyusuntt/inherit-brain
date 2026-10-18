---
title: "A company brain on FalkorDB, tested on 1,203 questions it had never answered"
published: false
description: "One graph per product, nine read-only Cypher tools, and a pre-registered test on Salesforce's HERB benchmark: 80.0 against 39.2 for retrieval on the same model."
tags: falkordb, wemakedevs, graphhacks, ai
cover_image: https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/cover.png
---

Day one on a new team. You inherit a product you have never seen: months of Slack channels, meeting transcripts, documents and pull requests. The people who knew how it works are busy or gone. Your first question is simple: who wrote the technical spec, and who reviewed it?

For Graph Hacks (WeMakeDevs x FalkorDB, Track 3: Company Brain) I built inherit-brain, an agent that answers that kind of question from a graph of everything the team left behind. This post is about what I built, where FalkorDB does the work, and what I learned by testing it on questions I had not seen.

Demo (every recorded answer, with its Cypher): https://zeyusuntt.github.io/inherit-brain/
Code: https://github.com/zeyusuntt/inherit-brain

## What it does

inherit-brain turns a product's records into one FalkorDB graph per product: people, documents, meetings, Slack messages, pull requests, bugs and their event histories, features, customers, and who did what to which. An agent answers questions by calling nine read-only tools. Each tool runs fixed, parameterised Cypher; the model never writes a query.

For the spec question on one product, SupportForce, the agent made one tool call (`doc_people`), which ran three read-only queries in 21.9 ms. It returned ten people, each with the record that puts them there. Two of them are different people who are both named Alice Davis, with different employee ids. The graph kept them apart because a person node is an employee id, not a name. HERB's own retrieval tools, on the same model, found one of the ten.

![The overview of the demo site](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/cover.png)

## How FalkorDB fits in

Each of the 30 products is its own graph, `herb:<product>`: 56,708 nodes and 164,698 edges in total, 16 node labels and 24 edge types. Every edge stores `src`, the id of the record it was read from, so an answer can always point back to a Slack message, a meeting line or a document.

The agent connects as a read-only user per product, `agent_<product>`, created with

```
ACL SETUSER agent_<p> reset on ><pw> (~herb:<p> +GRAPH.RO_QUERY)
```

so it can run `GRAPH.RO_QUERY` on its own product's graph and nothing else. A question about one product cannot read another.

Here is the main query behind one held-out question, "Find the employee IDs of engineers who authored maximum number of PRs in SecurityForce for bug fixes that are now reopened by customers":

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

It joins the bug, the PR that fixed it, the PR's author and the bug's history (resolved by that PR, then reopened). The tool counts fix PRs per author and keeps whoever is tied at the top. It returned Julia Miller, which is HERB's gold answer. Retrieval on the same model named four other engineers.

A full-text index (`db.idx.fulltext.queryNodes`) resolves titles and names that do not match exactly. A user-defined function, `herb_v1.statusAt(events, $as_of)`, gives a bug's status on any date from its event history. `algo.betweenness` on the collaboration edges ranks people who connect teams.

For "if this person leaves", a separate what-if user may use GRAPH.COPY to copy its product's graph into a scratch namespace, delete a person in the copy, and diff who owns the open bugs and PRs before and after. On SupportForce, if Julia Smith leaves, two open bugs and one open pull request have no owner left. An independent Python computation agreed with the graph on every case I checked.

![The what-if page: who leaves, and which open work loses its only owner](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/whatif.png)

Everything runs on FalkorDB 4.22.1 and 6.0.2; the tool tests run against both servers.

## Testing it so I could not fool myself

A demo is easy to cherry-pick, so I tested on a benchmark I did not write: Salesforce AI Research's [HERB](https://github.com/SalesforceAIResearch/HERB), a simulated company with 30 products, their records, and questions with gold answers (including questions that have no answer in the records). I built on six products and held back the other 24, with 1,203 questions.

Before the held-out run I froze the code, the prompts and the graphs (each hashed), wrote the plan down with the result I expected and what I would say for every outcome, and posted the plan's sha256 publicly (a [gist](https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67) and a [Wayback snapshot](https://web.archive.org/web/20261017025021/https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67)). A preflight refused to start any run whose hashes had drifted.

Eight ways to answer, all on the same model (Qwen3-30B-A3B on Nebius): inherit-brain; HERB's own retrieval tools in the same agent loop; the whole product in one prompt; HERB's hybrid retrieval; depth-1 access to entity cards, with and without a chain; retrieval over the same cards; and FalkorDB's GraphRAG-SDK with its automatic ontology. The whole project, extraction and grading included, spent 42.72 USD of a 50 USD cap.

| Same model | Score | Median input tokens |
|---|---|---|
| inherit-brain | 80.0 | 10,117 |
| HERB's retrieval tools, same loop | 39.2 | 46,451 |
| Whole product in one prompt | 26.0 | 125,947 |

The score is HERB's set F1 on the answer ids, averaged over person, company and PR or URL questions. Swapping the graph tools for retrieval, with everything else the same, cost 40.8 points (95% interval 35.7 to 46.2, product-clustered bootstrap). That is the hackathon's "removal test", measured.

![The removal test on the demo site: same model, same loop, graph tools against retrieval tools](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/removal.png)

## What I learned

The graph only helps when the tools do the hops. One arm gave the model typed, depth-1 access to the same entity cards: look up a node, list its neighbours. It did worse than plain retrieval over those cards (-13.8 points) and ran out of input budget on 68.9% of questions. Walking a graph one step per model call is slow and expensive. Multi-hop joins and counting belong inside Cypher, where one query does what ten model turns could not.

My tools fit the questions I saw. 610 of the held-out questions share their wording (with the product name swapped) with a development question. On those, inherit-brain led retrieval by 57.9 points. On the 593 never seen, by 20.1 (12.9 to 26.5). That is still a clear lead, about a third of the size, and the results page shows it next to the headline.

Before the held-out run, two model auditors (Codex and a Claude agent) checked sampled facts against the raw records, and the extraction failed five rounds. Bug histories needed exactly one assignment event per hand-over, on the message that makes it; review relations had swept in acknowledgements and work allocation that are not reviews; the definition of "key reviewer" itself had to be amended after round four. Each round got a fresh sample, and the failed rounds are published with the pre-registration.

A few FalkorDB details I did not expect:

- `%R~herb:p` ACL rules do not work for graph commands: `GRAPH.RO_QUERY` declares its key as read-write, so a read-only key rule never matches. A command-scoped selector, `(~herb:p +GRAPH.RO_QUERY)`, does.
- For the same reason, a `GRAPH.COPY` rule cannot say "this key is only a source", so the what-if user's limits are pinned by tests.
- The server's `TIMEOUT` was 1000 ms, so whole-graph identity reads (hashing a graph for the freeze) timed out until I passed a per-query timeout.
- `algo.betweenness` matched my hand-worked fixtures on 6.0.2 (5 of 5) but not on 4.22.1 (2 of 5), so 4.22 uses a Python fallback.

I wrote these up with minimal reproductions in [docs/sponsor-feedback.md](https://github.com/zeyusuntt/inherit-brain/blob/main/docs/sponsor-feedback.md).

Long context breaks on format before it breaks on facts. Reading the whole product in one prompt (about 126,000 tokens) failed on 49 questions, mostly invalid JSON from a repetition loop, and gave customer ids instead of names in 49 of 104 company answers. Mapping those ids to names moves it from 26.0 to 37.3. That is still far behind.

## Try it

The [explorer](https://zeyusuntt.github.io/inherit-brain/explore.html) shows all 1,203 held-out questions: the question, HERB's gold answer, every arm's answer and score, and inherit-brain's tool calls with the Cypher they ran. Questions it got wrong are there too.

![One held-out question in the explorer: gold answer, every arm's answer, and the tool path with its Cypher](https://raw.githubusercontent.com/zeyusuntt/inherit-brain/main/docs/blog/explorer.png)

To query the real graphs on your machine without a model key:

```bash
uv sync
docker compose up -d
make quickstart
uv run python -m inherit.ask --replay runs/eval/A5-held --slot SecurityForce/ans/21
```

The same nine tools are available as an MCP server (`uv run python -m inherit.mcp_server --product SecurityForce`).

I built this with Claude Code and Codex as my coding tools; the benchmark choice, the experiment design, the gates and the release decisions were mine. The README has the details.

When you inherit a product, you inherit its history. inherit-brain lets you ask it.
