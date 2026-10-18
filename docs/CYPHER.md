# Cypher in inherit-brain

The agent never writes Cypher. It picks one of nine typed tools, and each tool runs fixed, parameterised Cypher through `GRAPH.RO_QUERY` as the read-only ACL user `agent_<product>` (rule `(~herb:<product> +GRAPH.RO_QUERY)`). There is one graph per product, `herb:<product>`, 30 products in all. Values travel as query parameters; enums only choose among constant query fragments, and every variable-length bound is a literal ([catalog.py](../src/inherit/tools/catalog.py#L3)). The tool tests run against FalkorDB 4.22.1 and 6.0.2 ([falkor-versions.md](falkor-versions.md)). A tenth tool, `if_leaves`, is the what-if: it is not in the agent's spec (`agent._Full.specs` keeps only `kind == "readonly"`, [agent.py](../src/inherit/agent.py#L183)) and is not scored. It is covered in the last section.

Every query below is copied from the source. Where Python builds the text from fragments, the fragments are shown and the joined text is given as it was recorded. The recorded examples come from the held-out run in `site/data/q/<product>.json` (field `a5.path`: tool, params, rows, ms, and the exact queries run, on the 6.0 server). `ms` is the sum of recorded trace-operation timings assigned to that call, rounded to one decimal place; it excludes processing outside those operations. Each query also has its own timing.

## Schema at a glance

Node labels and the domain-key property in the extracted facts ([ontology.py](../src/inherit/ontology.py#L10)). In FalkorDB, every node's `id` is the qualified `<Label>:<key>` value; an original `id` property is stored as `source_id` ([load.py](../src/inherit/graph/load.py#L69)).

| Label | Key property |
|---|---|
| Product | name |
| Person | eid |
| Customer, Channel, Message, Document, Meeting, Bug, BugEvent, Feature, Observation, FeatureEvent | id |
| PR, Url, ExternalBug | link |
| Competitor | name |

Edge types and their allowed endpoints ([ontology.py](../src/inherit/ontology.py#L32)). Every edge is checked against this table before load.

| Edge | Endpoints |
|---|---|
| AUTHORED, SUGGESTED | Person to Document |
| ATTENDED, SPOKE_IN | Person to Meeting |
| POSTED | Person to Message |
| IN_CHANNEL | Message to Channel |
| MENTIONS | Message to Person |
| OPENED, REVIEWED | Person to PR |
| SHARED | Person to Url, Document or PR |
| COLLABORATED | Person to Person, stored once per unordered pair with src_id < dst_id |
| HAS | Product to Bug, Document, Meeting, Feature, PR or Competitor |
| REVERTS | PR to PR (revert to original), props `reason`, `applied` |
| ABOUT | PR to Feature; Url to Competitor or Product; Observation to Competitor, Product, Feature, Bug or Person; Message to PR |
| STATED | Person to Observation |
| REPORTED_BY | Bug to Customer; Observation to Customer or Person |
| CONTRIBUTED | Person to Feature |
| HAS_EVENT | Bug to BugEvent; Feature to FeatureEvent |
| IMPLEMENTED_BY | Feature to PR |
| DISCUSSES | Meeting or Message to Feature or Document |
| ASSIGNED_TO, ACTED_BY | BugEvent to Person |
| FIXES | PR to Bug |
| SIMILAR_TO | ExternalBug to Bug |

Edges carry a `src` list of record ids; tools return it as `evidence`. The loader writes nodes and edges with `UNWIND $rows AS r CREATE ... SET n = r.props` and `UNWIND $rows AS r MATCH (a:<L> {id: r.s}), (b:<L> {id: r.d}) CREATE (a)-[e:<TYPE>]->(b) SET e = r.props` ([load.py](../src/inherit/graph/load.py#L147)).

Indexes, constraints and the UDF are created by [schema.py](../src/inherit/graph/schema.py) during [load_product](../src/inherit/graph/load.py#L183):

| Kind | DDL | Where |
|---|---|---|
| Range index | `CREATE INDEX FOR (n:<Label>) ON (n.id)` for every label present, plus `Person.eid`, `PR.link`, `Bug.id` | [schema.py](../src/inherit/graph/schema.py#L37) |
| Constraints | `GRAPH.CONSTRAINT CREATE <graph> UNIQUE NODE <Label> PROPERTIES 1 <prop>` for Person.eid, Bug.id, PR.link; `MANDATORY` for Bug.status | [schema.py](../src/inherit/graph/schema.py#L44) |
| Full-text | `CREATE FULLTEXT INDEX FOR (n:<Label>) ON (n.<prop>)` for Person.name, Customer.company, Bug.title, PR.title, Document.type, Feature.title, Competitor.name | [schema.py](../src/inherit/graph/schema.py#L51) |
| Vector | `CREATE VECTOR INDEX FOR (n:<Label>) ON (n.embedding) OPTIONS {dimension:1024, similarityFunction:'cosine'}` for Message and Document | [schema.py](../src/inherit/graph/schema.py#L56) |
| UDF | library `herb_v1`, function `statusAt` | below |

Indexes and constraints are waited on with `CALL db.indexes()` and `CALL db.constraints()` until every row is OPERATIONAL ([schema.py](../src/inherit/graph/schema.py#L25)). Embeddings are written with `n.embedding = vecf32(r.emb)` ([load.py](../src/inherit/graph/load.py#L157)). No query in `catalog.py` or `whatif.py` reads the vector index. The nine tools use scans, range lookups, full-text search, traversals and betweenness, with Python filtering and aggregation.

The UDF source is [herb.js](../src/inherit/graph/udf/herb.js), loaded once with the client call `db.udf_load("herb_v1", source)` (never REPLACE; if the library exists its source must be identical or the load raises) ([schema.py](../src/inherit/graph/schema.py#L61)):

```js
// herb_v1.statusAt(events, date): status after the last event with ts <= date.
// events: list of {type, ts}; type in REPORTED, ASSIGNED, RESOLVED, DISMISSED, REOPENED.
function statusAt(events, date) {
    var best = null;
    var status = null;
    for (var i = 0; i < events.length; i++) {
        var e = events[i];
        if (e.ts <= date && (best === null || e.ts >= best)) {
            best = e.ts;
            if (e.type === 'RESOLVED') status = 'resolved';
            else if (e.type === 'DISMISSED') status = 'dismissed';
            else if (e.type === 'REPORTED' || e.type === 'ASSIGNED' || e.type === 'REOPENED') status = 'active';
        }
    }
    return status;
}
falkor.register('statusAt', statusAt);
```

A Python reference with the same rules, used to test the UDF, is [udf_ref.py](../src/inherit/graph/udf_ref.py). The ACL probe shows `GRAPH.UDF LOAD` refused for an agent-style user ([falkor-versions.md](falkor-versions.md#L49)), so only the build path loads it.

## Shared lookups

Several tools first resolve question text to ids with whole-label scans, then filter in Python. These are the fixed queries ([catalog.py](../src/inherit/tools/catalog.py#L209)):

```cypher
MATCH (p:Product) RETURN p.id, p.name, p.aliases, p.previous_names
MATCH (c:Customer) RETURN c.id, c.source_id, c.company
MATCH (b:Bug) WHERE b.component IS NOT NULL RETURN DISTINCT b.component
MATCH (p:Person) RETURN p.eid, p.name
MATCH (b:Bug) RETURN b.id, b.title, b.aliases
MATCH (f:Feature) RETURN f.id, f.title, f.topic, f.aliases, f.details
MATCH (d:Document) RETURN d.id, d.source_id, d.type, d.release_scope
```

(lines [220](../src/inherit/tools/catalog.py#L220), [233](../src/inherit/tools/catalog.py#L233), [244](../src/inherit/tools/catalog.py#L244), [266](../src/inherit/tools/catalog.py#L266), [274](../src/inherit/tools/catalog.py#L274), [306](../src/inherit/tools/catalog.py#L306), [342](../src/inherit/tools/catalog.py#L342).) The first one is why the product's own name is ignored as a filter: the graph is already one product. Results are paged after aggregation: at most 200 rows and 12,000 tokens (chars / 4) per response ([depth1.py](../src/inherit/tools/depth1.py#L36)), with an opaque cursor bound to product, tool and filters ([catalog.py](../src/inherit/tools/catalog.py#L159)).

## resolve_person

Maps a person's name to employee ids; namesakes are all returned with `ambiguous=true` ([tool spec](../src/inherit/tools/catalog.py#L1106), [impl](../src/inherit/tools/catalog.py#L354)). Without a meeting it reads all Person nodes and matches the folded name or eid in Python:

```cypher
MATCH (p:Person) RETURN p.eid, p.name, p.role, p.in_directory
```

With `meeting`, it reads the transcript-speaker edges of one meeting ([catalog.py](../src/inherit/tools/catalog.py#L359)):

```cypher
MATCH (p:Person)-[s:SPOKE_IN]->(m:Meeting) WHERE m.source_id = $meeting RETURN p.eid, p.name, p.role, p.in_directory, s.speaker_name, s.ambiguous, s.n_candidates, s.src
```

Parameters: `name` (required), `meeting` (optional, a `Meeting:` prefix is stripped).

| Question | Params | Rows | ms |
|---|---|---|---|
| FeedbackForce/ans/6: "What new features about integration with enterprise systems are added to FeedbackForce?" | `{"name": "Alice Brown"}` | 1 (the scan returned 216 Person rows) | 13.0 |

That slot is flagged `ops_assignment_ambiguous` in the recording (consecutive calls of the same tool); the displayed sum matches the single 13.0 ms query.

## resolve_entity

Resolves a document, customer, PR, bug, feature, competitor or product name to node ids: exact match first, then a full-text fallback for multi-word text ([spec](../src/inherit/tools/catalog.py#L1113), [impl](../src/inherit/tools/catalog.py#L386)). The Cypher is built from two constant tables. `KINDS` ([catalog.py](../src/inherit/tools/catalog.py#L379)) maps a kind to a label and the properties to match; the label and the property names are interpolated from that table, never from model text.

Exact pass (template, one query per requested kind, [catalog.py](../src/inherit/tools/catalog.py#L413)):

```cypher
MATCH (n:{label}) RETURN n.id, {cards._name_cols('n')}, n.{field0} AS m0, n.{field1} AS m1, ...
```

`{label}` is one of Customer, Document, Bug, PR, Feature, Competitor, Product. `_name_cols('n')` expands to the display-name columns, as recorded: `n.name AS nm0, n.company AS nm1, n.title AS nm2, n.type AS nm3, n.document_type AS nm4, n.source_id AS nm5, n.link AS nm6, n.eid AS nm7` ([cards.py](../src/inherit/graph/cards.py#L101)).

Full-text fallback, only when the exact pass found nothing and the text has two or more words, and only for labels in `FALLBACK_LABELS` (Bug, Feature, Document, Competitor; [depth1.py](../src/inherit/tools/depth1.py#L39)) ([catalog.py](../src/inherit/tools/catalog.py#L427)):

```cypher
CALL db.idx.fulltext.queryNodes('{label}', $q) YIELD node, score RETURN node.id, score, node.name AS nm0, node.company AS nm1, node.title AS nm2, node.type AS nm3, node.document_type AS nm4, node.source_id AS nm5, node.link AS nm6, node.eid AS nm7
```

`$q` is normalised text with every character other than a letter, number, whitespace or underscore escaped (`_escape`, [depth1.py](../src/inherit/tools/depth1.py#L74)). Scores are sorted in Python, ties at the cut stay. The held-out recordings contain 79 such full-text queries on `Feature`, 8 on `Bug` and 1 on `Document`.

Parameters: `text` (required), `kinds` (optional list; `component` returns distinct `Bug.component` values via the shared query above).

| Question | Params | Rows | ms |
|---|---|---|---|
| PersonalizeForce/ans/14: "Can you provide links to PRs for the integration of Shopify and Salesforce in PersonalizeForce that were not approved?" | `{"text": "Shopify and Salesforce integration", "kinds": ["feature"]}` | 1 (fulltext query returned 1, in 6.6 ms; Feature scan 59 rows) | 22.1 |

## doc_people

Authors and key reviewers of a document type ([spec](../src/inherit/tools/catalog.py#L1124), [impl](../src/inherit/tools/catalog.py#L451)). After the document lookup (shared query above), two queries ([catalog.py](../src/inherit/tools/catalog.py#L456)):

```cypher
MATCH (p:Person)-[au:AUTHORED]->(d:Document) WHERE d.id IN $docs AND ($role IS NULL OR p.role = $role) RETURN p.eid, p.name, p.role, d.id, au.src
MATCH (p:Person)-[s:SUGGESTED]->(d:Document) WHERE d.id IN $docs AND s.kind IN ['suggestion', 'review_question'] AND ($role IS NULL OR p.role = $role) RETURN p.eid, p.name, p.role, d.id, s.kind, s.section, s.text, s.ambiguous, s.src
```

With `return=suggestions` and no role, one more query adds anonymous feedback ([catalog.py](../src/inherit/tools/catalog.py#L469)):

```cypher
MATCH (d:Document) WHERE d.id IN $docs AND d.feedback_items IS NOT NULL RETURN d.id, d.source_id, d.feedback_items
```

Parameters: `document_type` (required), `scope` (any, current, previous), `role`, `return` (people, suggestions). Feature: `IN` list parameters and `$role IS NULL OR` optional filters. A review question counts as substantive only if every source record is a Slack message id; that rule is applied in Python.

| Question | Params | Rows | ms |
|---|---|---|---|
| SupportForce/ans/3: "Find employee IDs of the authors and key reviewers of the Technical Specifications Document for the SupportForce product?" | `{"document_type": "Technical Specifications Document", "return": "people"}` | 10 (authors query 3 rows, suggestions query 16) | 21.9 |

## bug_board

Counts bugs by company or by person ([spec](../src/inherit/tools/catalog.py#L1138), [impl](../src/inherit/tools/catalog.py#L519)). This tool builds its Cypher from fragments; what varies is the status predicate, the optional `as_of` selector, and the actor branch.

Selector fragments ([catalog.py](../src/inherit/tools/catalog.py#L496)). `{status}` is one of `b.unresolved = true`, `b.resolved = true`, `b.active = true`, `b.reopened = true`, `b.no_fix_needed = true`, or `true` (for `any`):

```cypher
MATCH (b:Bug) WHERE {status} AND ($components IS NULL OR b.component IN $components) AND ($companies IS NULL OR any(x IN b.companies WHERE x IN $companies))
```

With `as_of`, the selector is instead the one below, and `@AT` becomes `= $at_status` (status active, resolved or no_fix_needed mapped to active, resolved, dismissed) or `IS NOT NULL` (status any). This is where the UDF is called:

```cypher
MATCH (b:Bug)-[:HAS_EVENT]->(ev:BugEvent) WITH b, collect({type: ev.type, ts: ev.ts}) AS evs WHERE herb_v1.statusAt(evs, $as_of) @AT AND ($components IS NULL OR b.component IN $components) AND ($companies IS NULL OR any(x IN b.companies WHERE x IN $companies))
```

Suffix appended per actor ([catalog.py](../src/inherit/tools/catalog.py#L534)):

```cypher
 MATCH (b)-[rb:REPORTED_BY]->(c:Customer) WHERE c.company IS NOT NULL AND ($companies IS NULL OR c.company IN $companies) RETURN c.company, c.id, b.id, rb.src
```
(actor `company`)

```cypher
 MATCH (b)-[he:HAS_EVENT]->(ev:BugEvent)-[at:ASSIGNED_TO]->(p:Person) WHERE ev.type = 'ASSIGNED' AND p.eid IN b.current_assignees RETURN p.eid, p.name, p.role, b.id, null, at.src, he.src, [ev.id]
```
(actor `assignee`)

```cypher
 MATCH (b)-[he:HAS_EVENT]->(ev:BugEvent)-[ab:ACTED_BY]->(p:Person) WHERE ev.type = $etype RETURN p.eid, p.name, p.role, b.id, null, ab.src, he.src, [ev.id]
```
(actors `resolver` and `dismisser`, `$etype` is `RESOLVED` or `DISMISSED`; with `status=reopened` this is followed by ` MATCH (b)-[:HAS_EVENT]->(ro:BugEvent) WHERE ro.type = 'REOPENED' AND ro.ts > ev.ts` before the RETURN)

```cypher
 MATCH (pr:PR)-[fx:FIXES]->(b) MATCH (p:Person)-[op:OPENED]->(pr) RETURN p.eid, p.name, p.role, b.id, pr.link, op.src, fx.src, []
```
(actor `fix_pr_author`; with `status=reopened` the join below is used, see the example)

`return=bugs` uses the selector plus ([catalog.py](../src/inherit/tools/catalog.py#L621)):

```cypher
 RETURN b.id, b.title, b.impact_text, b.component, b.companies, b.unresolved, b.resolved, b.active, b.reopened, b.no_fix_needed
```

Support records for the returned bugs come from two more queries ([catalog.py](../src/inherit/tools/catalog.py#L587)):

```cypher
MATCH (b:Bug)-[he:HAS_EVENT]->(ev:BugEvent) WHERE b.id IN $ids RETURN b.id, ev.id, ev.type, ev.ts, he.src
MATCH (pr:PR)-[fx:FIXES]->(b:Bug) WHERE b.id IN $ids RETURN b.id, pr.link, fx.src
```

plus `MATCH (b:Bug)-[rb:REPORTED_BY]->(c:Customer) WHERE b.id IN $ids RETURN b.id, c.id, rb.src` for `return=bugs` ([catalog.py](../src/inherit/tools/catalog.py#L626)). Grouping, counting distinct bugs (or distinct PR links for `fix_pr_author`) and `rank=max` ties are done in Python after the query.

Parameters: `status`, `customer`, `component`, `actor`, `rank`, `as_of`, `return`.

| Question | Params | Rows | ms |
|---|---|---|---|
| SecurityForce/ans/21: "Find the employee IDs of engineers who authored maximum number of PRs in SecurityForce for bug fixes that are now reopened by customers." | `{"status": "reopened", "component": "SecurityForce", "actor": "fix_pr_author", "rank": "max", "return": "people"}` | 1 (main query 4 rows, 13.0 ms; events query 20 rows) | 17.2 |

The recorded main query (`component` equals the product name, so `$components` was null):

```cypher
MATCH (b:Bug) WHERE b.reopened = true AND ($components IS NULL OR b.component IN $components) AND ($companies IS NULL OR any(x IN b.companies WHERE x IN $companies)) MATCH (pr:PR)-[fx:FIXES]->(b) MATCH (p:Person)-[op:OPENED]->(pr) MATCH (b)-[hs:HAS_EVENT]->(rs:BugEvent) WHERE rs.type = 'RESOLVED' AND pr.link IN rs.fix_pr_links MATCH (b)-[:HAS_EVENT]->(ro:BugEvent) WHERE ro.type = 'REOPENED' AND ro.ts > rs.ts RETURN p.eid, p.name, p.role, b.id, pr.link, op.src, fx.src + hs.src, [rs.id]
```

No held-out recording used `as_of`, so `herb_v1.statusAt` has no recorded example; its behaviour is covered by the probe rows in [falkor-versions.md](falkor-versions.md#L127).

## fix_trail

Customer company, its bugs, the PRs that fix them, the PR authors ([spec](../src/inherit/tools/catalog.py#L1165), [impl](../src/inherit/tools/catalog.py#L644)). One fixed query, a three-hop pattern with optional list filters, preceded by the product, bug (and customer, person) lookups from the shared section ([catalog.py](../src/inherit/tools/catalog.py#L645)):

```cypher
MATCH (pr:PR)-[fx:FIXES]->(b:Bug)-[rb:REPORTED_BY]->(c:Customer) WHERE ($companies IS NULL OR c.company IN $companies) AND ($bugs IS NULL OR b.id IN $bugs) MATCH (p:Person)-[op:OPENED]->(pr) WHERE ($engineers IS NULL OR p.eid IN $engineers) RETURN pr.link, pr.title, b.id, b.title, c.id, c.company, p.eid, p.name, fx.src, op.src, rb.src
```

Parameters: `customer`, `bug`, `engineer` (all optional). `$companies` contains company names, `$bugs` contains Bug node ids, and `$engineers` contains employee ids; each is null when its filter is omitted.

| Question | Params | Rows | ms |
|---|---|---|---|
| PitchForce/ans/25 (step 5): "Find the employee ID of engineers who resolved issues with the AI Model & Personalization Engine in PitchForce that were causing pitches to fail in resonating with target audiences and damaging customer relationships?" | `{"customer": "PitchForce", "bug": "Bug:PitchForce:ch-force-87f4a7:feature-extraction-failure"}` | 1 | 6.9 |

The recorded run bound `{"companies": null, "bugs": ["Bug:PitchForce:ch-force-87f4a7:feature-extraction-failure"], "engineers": null}` (the customer equalled the product name and was dropped); the main query took 2.3 ms and returned 1 row.

## feature_fate

Features with dated history, selected by derived fate (implemented, unimplemented, reverted, delayed_then_added, held, any) ([spec](../src/inherit/tools/catalog.py#L1173), [impl](../src/inherit/tools/catalog.py#L780)). It reads whole feature-scoped relations with six fixed queries and joins them in Python; the fate flags are stored Feature properties, not computed in Cypher.

```cypher
MATCH (f:Feature) RETURN f.id, f.title, f.aliases, f.details, f.implemented, f.reverted, f.held, f.delayed_then_added, f.discussed_not_implemented
MATCH (f:Feature)-[:HAS_EVENT]->(ev:FeatureEvent) RETURN f.id, ev.id, ev.kind, ev.date, ev.reason, ev.reason_text, ev.record_id, ev.pr_links
MATCH (f:Feature)-[ib:IMPLEMENTED_BY]->(pr:PR) OPTIONAL MATCH (rv:PR)-[r:REVERTS]->(pr) RETURN f.id, pr.link, ib.src, rv.link, r.reason, r.applied, r.src
MATCH (pr:PR)-[a:ABOUT]->(f:Feature) WHERE pr.repo_scope = 'product' RETURN f.id, pr.link, pr.title, a.src, pr.merged
MATCH (rv:PR)-[r:REVERTS]->(pr:PR) RETURN pr.link, rv.link, r.reason, r.applied, r.src
```

(lines [786](../src/inherit/tools/catalog.py#L786), [789](../src/inherit/tools/catalog.py#L789), [795](../src/inherit/tools/catalog.py#L795), [800](../src/inherit/tools/catalog.py#L800), [804](../src/inherit/tools/catalog.py#L804).) For `return=people` one more query runs ([catalog.py](../src/inherit/tools/catalog.py#L876)):

```cypher
MATCH (p:Person)-[c:CONTRIBUTED]->(f:Feature) WHERE f.id IN $ids RETURN f.id, p.eid, p.name, p.role, c.kind, c.ambiguous, c.src
```

When a feature phrase's best match has no rows, `_with_alternatives` re-runs the same queries for up to five other features whose names contain the phrase tokens and puts the counts in a hint ([catalog.py](../src/inherit/tools/catalog.py#L737)); that is why recordings show 16 to 23 queries for some calls.

Parameters: `topic`, `fate`, `reason` (security, product_direction, introduced_bugs, other), `return` (features, people, prs).

| Question | Params | Rows | ms |
|---|---|---|---|
| CollaborateForce/ans/22: "Can you find the links to the PRs for the WebRTC optimization in CollaborateForce that need to be merged?" | `{"topic": "WebRTC optimization", "fate": "implemented", "return": "prs"}` | 2 (feature scan 56 rows, events 144, IMPLEMENTED_BY 16, ABOUT 40, REVERTS 5) | 28.2 |

## who_said

Typed observations and demo links ([spec](../src/inherit/tools/catalog.py#L1194), [impl](../src/inherit/tools/catalog.py#L911)). The match clause is chosen by `subject` from constant tables ([catalog.py](../src/inherit/tools/catalog.py#L897)); `$kinds` comes from a constant map of `(subject, kind)`:

```cypher
MATCH (o:Observation)-[:ABOUT]->(:Competitor) WHERE o.kind IN $kinds
MATCH (o:Observation) WHERE o.kind IN $kinds AND o.release_scope = 'previous'
MATCH (o:Observation)-[:ABOUT]->(:Product) WHERE o.kind IN $kinds AND coalesce(o.release_scope, 'unknown') <> 'previous'
```

(subjects `competitor`, `previous_release`, `product`), followed by ` RETURN o.id, o.kind, o.text, o.release_scope, o.record_id` and these lookups by observation id ([catalog.py](../src/inherit/tools/catalog.py#L935)):

```cypher
MATCH (p:Person)-[s:STATED]->(o:Observation) WHERE o.id IN $ids RETURN o.id, p.eid, p.name, p.role, s.src
MATCH (o:Observation)-[rb:REPORTED_BY]->(c:Customer) WHERE o.id IN $ids RETURN o.id, c.id, c.company, rb.src
MATCH (o:Observation)-[rb:REPORTED_BY]->(p:Person) WHERE o.id IN $ids RETURN o.id, p.eid, p.name, p.role, rb.src
MATCH (o:Observation)-[:ABOUT]->(c:Competitor) WHERE o.id IN $ids RETURN o.id, c.name
```

For `kind=demo` ([catalog.py](../src/inherit/tools/catalog.py#L903), [982](../src/inherit/tools/catalog.py#L982)), the match is one of the following, followed by ` RETURN u.link`, then the sharers:

```cypher
MATCH (u:Url)-[:ABOUT]->(:Competitor) WHERE u.kind = 'demo'
MATCH (u:Url)-[:ABOUT]->(:Product) WHERE u.kind = 'demo' AND u.release_scope = 'previous'
MATCH (u:Url)-[:ABOUT]->(:Product) WHERE u.kind = 'demo' AND coalesce(u.release_scope, 'unknown') <> 'previous'
MATCH (p:Person)-[sh:SHARED]->(u:Url) WHERE u.link IN $links RETURN u.link, p.eid, p.name, p.role, sh.src
```

For `kind=worked_on` (previous release only, [catalog.py](../src/inherit/tools/catalog.py#L1018)):

```cypher
MATCH (p:Person)-[au:AUTHORED]->(d:Document) WHERE d.release_scope = 'previous' AND ($role IS NULL OR p.role = $role) RETURN p.eid, p.name, p.role, d.id, au.src
MATCH (o:Observation)-[rb:REPORTED_BY]->(p:Person) WHERE o.kind = 'qa_issue' AND o.release_scope = 'previous' AND ($role IS NULL OR p.role = $role) RETURN p.eid, p.name, p.role, o.id, rb.src
```

Parameters: `subject` (required), `kind`, `role`, `return` (people, claims, companies, urls).

| Question | Params | Rows | ms |
|---|---|---|---|
| EdgeForce/ans/0: "Find employee IDs of Marketing Research Analysts who worked on the previous release of EdgeForce?" | `{"subject": "previous_release", "kind": "worked_on", "role": "Marketing Research Analyst", "return": "people"}` | 2 (authors query 2 rows, 13.9 ms; QA reporters query 0 rows, 9.7 ms) | 23.6 |

## pr_board

Product-repository PRs by approval, state, kind and feature ([spec](../src/inherit/tools/catalog.py#L1207), [impl](../src/inherit/tools/catalog.py#L669)). The main query is assembled from `match` and `where` fragment lists ([catalog.py](../src/inherit/tools/catalog.py#L673)):

- first match: `MATCH (pr:PR)`, or `MATCH (rv:PR)-[r:REVERTS]->(pr:PR)` when `state=reverted`;
- added match when a feature is given or `kind=feature_development`: `MATCH (pr)-[:ABOUT]->(f:Feature)`;
- always: `pr.repo_scope = 'product'`;
- optional predicates joined with AND: `($revert_reason IS NULL OR r.reason = $revert_reason)`, `pr.approval = $approval`, `pr.pending_action = 'need_merge'`, `pr.pending_action = 'need_close'`, `pr.merged = true`, `f.id IN $features`, `NOT (pr)-[:FIXES]->(:Bug)` (feature_development), `(pr)-[:FIXES]->(:Bug)` (bug_fix).

The text is `<matches> WHERE <predicates> RETURN DISTINCT pr.link, pr.title, pr.approval, pr.merged, pr.state, pr.pending_action` ([catalog.py](../src/inherit/tools/catalog.py#L690)). Then three lookups by link ([catalog.py](../src/inherit/tools/catalog.py#L700)):

```cypher
MATCH (p:Person)-[op:OPENED]->(pr:PR) WHERE pr.link IN $links RETURN pr.link, p.eid, p.name, p.role, op.src
MATCH (pr:PR)-[a:ABOUT]->(f:Feature) WHERE pr.link IN $links RETURN pr.link, f.id, f.title, a.src
MATCH (rv:PR)-[r:REVERTS]->(pr:PR) WHERE pr.link IN $links RETURN pr.link, rv.link, r.reason, r.applied, r.src
```

Parameters: `feature`, `approval`, `state`, `revert_reason`, `kind`, `rank`, `return`. With `return=people`, authors are grouped over distinct PR links, so feature and review fan-out never inflates a count.

| Question | Params | Rows | ms |
|---|---|---|---|
| CoachForce/ans/13: "Find the employee ID of engineer with the highest number of approved feature development PRs in CoachForce?" | `{"kind": "feature_development", "approval": "approved", "return": "people", "rank": "max"}` | 1 (main query 17 rows, 10.5 ms) | 14.9 |

Recorded main query:

```cypher
MATCH (pr:PR) MATCH (pr)-[:ABOUT]->(f:Feature) WHERE pr.repo_scope = 'product' AND pr.approval = $approval AND NOT (pr)-[:FIXES]->(:Bug) RETURN DISTINCT pr.link, pr.title, pr.approval, pr.merged, pr.state, pr.pending_action
```

## bridges

Experimental directed betweenness on the stored collaboration projection. Each unordered collaboration is stored once in ID order, so this ranking does not establish who is a knowledge bottleneck. FalkorDB 6.0 uses the native algorithm; 4.22.1 uses the Python fallback. No held-out question called this tool. ([spec](../src/inherit/tools/catalog.py#L1229), [impl](../src/inherit/tools/catalog.py#L1069)). On a 6.0 server it runs the native algorithm, with a fixed seed and a sample size above the number of people per product ([catalog.py](../src/inherit/tools/catalog.py#L1064)):

```cypher
CALL algo.betweenness({nodeLabels: ['Person'], relationshipTypes: ['COLLABORATED'], samplingSize: 1000, samplingSeed: 42}) YIELD node, score RETURN node.eid, node.name, node.role, score
```

On 4.22.1 the tool does not use the native call, because the probe shows wrong scores there ([falkor-versions.md](falkor-versions.md#L164): a 5-node directed fixture gives v3 and v4 2.0 instead of 1.0, and a chain with `samplingSize` above the node count scores 0). It reads the same projection and runs Brandes' algorithm in Python ([catalog.py](../src/inherit/tools/catalog.py#L1032)), with each row labelled `engine: "native"` or `"python"`:

```cypher
MATCH (p:Person) RETURN p.eid, p.name, p.role
MATCH (a:Person)-[:COLLABORATED]->(b:Person) RETURN a.eid, b.eid
```

Parameters: `top_k` (1 to 100, default 10). There is no recorded example.

## The what-if: if_leaves

`if_leaves(person)` answers "which open bugs and PRs lose their only recorded owner, and what depends on them" on a copy of the graph. It runs as `whatif_<product>`, never as the agent user ([whatif.py](../src/inherit/whatif.py#L1)). ACL rules ([acl.py](../src/inherit/graph/acl.py#L23)):

```
(~whatif:{p}:* ~herb:{p} +GRAPH.COPY)
(~whatif:{p}:* +GRAPH.QUERY +GRAPH.RO_QUERY +GRAPH.DELETE)
(~herb:{p} +GRAPH.RO_QUERY)
```

While `herb:<p>` exists, the user can read it, copy it into `whatif:<p>:*`, and read, write and delete those copies. Direct writes and deletes on the source are refused. COPY permissions do not distinguish source from destination, so the source must remain present while this user is live ([falkor-versions.md](falkor-versions.md#L279)). The probe confirms writes to `herb:<p>`, a copy into another product's key, and a delete of the source are refused, and that the copy is deleted in a `finally` ([falkor-versions.md](falkor-versions.md#L96)). The agent user has only `(~herb:{p} +GRAPH.RO_QUERY)` ([acl.py](../src/inherit/graph/acl.py#L19)).

Steps ([whatif.py](../src/inherit/whatif.py#L162)):

1. Check the person exists in the source (`GRAPH.RO_QUERY`): `MATCH (p:Person {id: $pid}) RETURN count(p)`.
2. `GRAPH.COPY herb:<p> whatif:<p>:<uuid4 hex>`. A copy forks the server, so a refusal containing "could not fork" is attempted up to 12 times, with a growing sleep after retryable refusals ([whatif.py](../src/inherit/whatif.py#L110)).
3. In the copy, materialise `RESPONSIBLE_FOR` (Person to item) and `DEPENDS_ON` (dependent to prerequisite) from the original edges, and mark features with an open development PR ([whatif.py](../src/inherit/whatif.py#L51)):

```cypher
MATCH (b:Bug)-[:HAS_EVENT]->(e:BugEvent) WHERE b.active = true AND e.type = 'ASSIGNED' WITH b, e ORDER BY e.ts DESC, coalesce(toInteger(split(e.message_id, '-')[0]), -1) DESC, coalesce(toInteger(split(e.message_id, '-')[1]), -1) DESC, coalesce(e.seq, 0) DESC, e.id DESC WITH b, collect(e.id)[0] AS last MATCH (l:BugEvent {id: last})-[:ASSIGNED_TO]->(p:Person) MERGE (p)-[:RESPONSIBLE_FOR]->(b)
MATCH (p:Person)-[:OPENED]->(r:PR) WHERE r.state = 'open' AND r.merged = false MERGE (p)-[:RESPONSIBLE_FOR]->(r)
MATCH (f:Feature)-[:IMPLEMENTED_BY]->(r:PR) MERGE (f)-[:DEPENDS_ON]->(r)
MATCH (r:PR)-[a:ABOUT]->(f:Feature) WHERE a.kind = 'development' MERGE (f)-[:DEPENDS_ON]->(r)
MATCH (r:PR)-[:FIXES]->(b:Bug) MERGE (b)-[:DEPENDS_ON]->(r)
MATCH (r:PR)-[x:ABOUT]->(f:Feature) WHERE x.kind = 'development' AND r.state = 'open' AND r.merged = false SET f.pending_dev = true
```

4. Group 1 (already unowned) and the set of owned items, before the delete ([whatif.py](../src/inherit/whatif.py#L71)):

```cypher
MATCH (w) WHERE ((w:Bug AND w.active = true) OR (w:PR AND w.state = 'open' AND w.merged = false)) AND NOT ()-[:RESPONSIBLE_FOR]->(w) RETURN w.id
MATCH (w) WHERE ((w:Bug AND w.active = true) OR (w:PR AND w.state = 'open' AND w.merged = false)) AND ()-[:RESPONSIBLE_FOR]->(w) RETURN w.id
```

5. The delete, in the copy only:

```cypher
MATCH (p:Person {id: $pid}) DETACH DELETE p
```

6. The diff. Group 2 (newly unowned) is the owned-before items that have no owner now:

```cypher
MATCH (w) WHERE ((w:Bug AND w.active = true) OR (w:PR AND w.state = 'open' AND w.merged = false)) AND w.id IN $owned AND NOT ()-[:RESPONSIBLE_FOR]->(w) RETURN w.id
```

Group 3 (affected) is a directed variable-length walk of 1 to 4 hops from the group-2 items back to their dependents. The bound is a literal (the code cites FalkorDB issue #3350 for parameterised bounds). `$excluded` is groups 1 and 2; `bound_hit` is true if a path of exactly 4 edges exists:

```cypher
MATCH (a)-[:DEPENDS_ON*1..4]->(b) WHERE b.id IN $newly AND NOT a.id IN $excluded AND (((a:Bug AND a.active = true) OR (a:PR AND a.state = 'open' AND a.merged = false)) OR (a:Feature AND (a.discussed_not_implemented = true OR (coalesce(a.implemented, false) = false AND a.pending_dev = true)))) RETURN DISTINCT a.id
MATCH (a)-[:DEPENDS_ON*4]->(b) WHERE b.id IN $newly AND NOT a.id IN $excluded AND (((a:Bug AND a.active = true) OR (a:PR AND a.state = 'open' AND a.merged = false)) OR (a:Feature AND (a.discussed_not_implemented = true OR (coalesce(a.implemented, false) = false AND a.pending_dev = true)))) RETURN a.id LIMIT 1
```

7. Cleanup, always, in `finally` ([whatif.py](../src/inherit/whatif.py#L142)): refuse to delete any key that does not start with `whatif:`, run `GRAPH.DELETE <copy>` if the key exists, then verify with `GRAPH.RO_QUERY <copy> "RETURN 1"` that it now fails with "empty key". If the delete fails while another error is in flight, the original error is kept and the cleanup failure is added as a note.

Recorded outcomes for the demo are in `site/data/whatif.json` (up to 3 Engineering Leads per product, plus 5 owners). Example: CoachForce, Julia Garcia (Engineering Lead): 0 already unowned, 2 newly unowned, 1 affected, `bound_hit` false, `max_hops` 4. The tool call time is not recorded there.

### Independent check

[whatif_check.py](../src/inherit/whatif_check.py#L42) recomputes the three groups and `bound_hit` in plain Python from an exported snapshot (nodes with stored properties, plus the original typed edges). It shares only the snapshot and the schema names with `whatif.py`; it never sees a projected edge, an owner set or a result row of the graph side. `compare` ([whatif_check.py](../src/inherit/whatif_check.py#L113)) runs both for every Engineering Lead of a product and reports any difference item by item:

```
uv run python -m inherit.whatif_check --products dev [--servers 4.22,6.0]
```

The recorded result is [results/freeze/whatif_check.json](../results/freeze/whatif_check.json): on each of the two servers, 30 products and 206 leads, 0 mismatches.

## Try it

Complete the graph-loading setup first; starting Docker alone does not create the product graphs or ACL users. `make quickstart` downloads the release assets and loads all 30 graphs, with no model key:

```
uv sync
docker compose up -d
make quickstart
```

Replay the recorded tool calls of one held-out slot against the live graph, with no model call. It prints each call, its Cypher, row count, and fresh timings next to the recorded ones:

```
uv run python -m inherit.ask --replay runs/eval/A5-held --slot SecurityForce/ans/21
```

The nine tools are also served over MCP (stdio) as the read-only `agent_<product>` user, one server per product; see [MCP.md](MCP.md):

```
uv run python -m inherit.mcp_server --product AutoTuneForce
```
