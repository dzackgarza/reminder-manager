set fallback := true

install:
    uv sync --dev

lint:
    .venv/bin/ruff check .

format:
    .venv/bin/ruff format .

typecheck:
    .venv/bin/basedpyright

test:
    .venv/bin/pytest

check: lint typecheck test
