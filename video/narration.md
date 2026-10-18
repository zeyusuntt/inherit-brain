# Narration

The video's narration, one paragraph per scene; `/` marks a pause of about one second. `video/scripts/timeline.py` reads it to align the recording and time the captions.

## The narration (draft 3, for recording)

How to read it: one take, the way you would explain it to a friend at a desk. Pause about a second at each `/`, and leave two seconds of silence between paragraphs (that is where the cuts go). If you stumble, pause and restart the sentence; the stumble gets cut. Every number is from the frozen run (results/RESULTS.md); the examples are SupportForce/ans/3 and the SupportForce what-if for Julia Smith.

> Day one on a new team. You inherit a product you have never seen: months of Slack, meetings, documents and pull requests. / Your first question is simple. Who wrote the technical spec, and who reviewed it? /
>
> Hi, I'm Zeyu. This is inherit-brain, a company brain on FalkorDB. It turns what a team left behind into one graph per product: people, bugs, documents, pull requests, and who did what to which. /
>
> Ask it, and it does not hunt through documents. It walks the graph. / Ten people wrote or reviewed that spec, including two different people named Alice Davis. The graph returns all ten, each with the record that puts them there. /
>
> A demo is easy to cherry-pick. So I tested it on Salesforce's HERB benchmark: thirty simulated products. I built on six, and held back twelve hundred questions from the other twenty-four. / Before the held-out run, I froze the code and posted the hash of my plan in public. /
>
> Eight ways to answer, all on the same model, Qwen3 on Nebius. They include HERB's own retrieval tools, reading the whole product at once, vector search, and FalkorDB's own GraphRAG SDK. /
>
> The graph scored eighty. HERB's tools in the same loop, thirty-nine. Reading everything, twenty-six, with twelve times the input. / Swap the graph tools for retrieval, and the same model loses forty points. / On wordings it never saw in development, the lead is smaller: twenty points. /
>
> It was not clean. Two AI auditors checked the extracted facts against the raw records, and they failed it five times. / Each time I fixed the extraction or the audit rules, drew a fresh sample, and kept the failures on the record. /
>
> FalkorDB does real work here: a separate graph per product, a read-only user for the agent, Cypher that joins a bug, its fix, its author and its history in one query, / and GRAPH.COPY for the uncomfortable question. If Julia Smith leaves, two open bugs and a pull request have no owner left. /
>
> You can check all of it. Every answer, every query and every score, next to the benchmark's own gold answer. / The link is below. /
>
> When you inherit a product, you inherit its history. inherit-brain lets you ask it. /

## How to say it

- Slowest: the first two sentences, "It walks the graph.", the three scores, and the last two sentences.
- "Alice Davis", "Julia Smith": clear and a little slower.
- Numbers flat, no emphasis: "eighty", "thirty-nine", "twenty-six", "forty", "twenty".
- "GRAPH.COPY" is said "graph copy". "HERB" is said like the word herb. "Qwen" is said "chwen".
