.PHONY: test lint format

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run ruff format .
