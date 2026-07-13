.PHONY: install data experiment test lint format api dashboard screenshot clean

PYTHON ?= .venv/bin/python

install:
	python3 -m venv .venv
	.venv/bin/pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e '.[dev]'

data:
	$(PYTHON) -m velacl.data --output data/streaming/events.jsonl

experiment: data
	$(PYTHON) -m velacl.experiment --config configs/base.yaml

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

api:
	$(PYTHON) -m uvicorn apps.api.main:app --reload

dashboard:
	cd apps/dashboard && npm install && npm run dev

screenshot:
	cd apps/dashboard && npx playwright install chromium && node scripts/screenshot.mjs

clean:
	rm -rf .pytest_cache .ruff_cache build dist apps/dashboard/.next
