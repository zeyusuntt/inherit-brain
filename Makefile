.PHONY: download bundle audit test proxy

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
