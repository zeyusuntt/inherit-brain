# Working in this repo

A company brain on FalkorDB over the Salesforce HERB benchmark, built for Graph Hacks (WeMakeDevs × FalkorDB), Track 3.

If `docs/internal/HANDOFF.md` exists, read it first, then `docs/internal/PLAN.md`. `docs/internal/` holds private notes and is git-ignored.

## Rules

1. Every model call goes through `inherit.llm.chat()`, which meters spend against the hard cap in `.env`. Never call a provider directly.
2. Only the allowlisted HERB fields enter the input bundle. Only `src/eval/questions.py` reads question files. The agent receives the question text and nothing else.
3. Raw HERB data stays in `data/` (git-ignored). Never commit it.
4. Tests first for every behaviour change; tool tests run against both FalkorDB 4.22.1 and 6.0.2 (`docker compose up -d`).
5. Do not push without Zeyu's yes. No AI attribution in commit messages.
6. Docs voice: plain, concrete, numbers over adjectives, no em-dashes.

## Run

```
uv sync
docker compose up -d
uv run pytest -q
```
