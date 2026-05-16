.PHONY: setup setup-cpu test lint fmt gpu-check smoke docs-status clean

setup:
	uv sync --extra cu129 --group dev

setup-cpu:
	uv sync --extra cpu --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check . --fix

gpu-check:
	uv run python scripts/check_gpu.py

smoke:
	uv run python scripts/smoke_imports.py

docs-status:
	@sed -n '1,200p' docs/status.md

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__
