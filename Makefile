VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

.PHONY: setup test lint format eval-small eval-full

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e "[dev]" || true
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

eval-small:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite dev --out artifacts

eval-full:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite full --out artifacts
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results.csv --out artifacts/plots
