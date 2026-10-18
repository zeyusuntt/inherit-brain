.PHONY: download bundle audit test proxy load fetch quickstart

download:
	uv run python -m inherit.data.download

bundle:
	uv run python -m inherit.data.bundle

audit:
	uv run python -m inherit.audit

test:
	uv run pytest -q

proxy:
	uv run python -m inherit.proxy

PRODUCTS ?= dev

load:
	uv run python -m inherit.graph.load --products $(PRODUCTS)

fetch:
	python3 scripts/fetch_release.py

quickstart: fetch
	$(MAKE) load PRODUCTS=all
