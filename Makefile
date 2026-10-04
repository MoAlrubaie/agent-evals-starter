# One-command entry points. `make demo` works from a fresh clone.

# First Python on PATH that is 3.10 or newer (override with `make PYTHON=python3.12 ...`).
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3.10 python3; do \
	command -v $$p >/dev/null 2>&1 && $$p -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null && echo $$p && break; \
	done)
VENV := .venv
BIN := $(VENV)/bin
INSTALLED := $(BIN)/agent-evals
# Run from src/ directly, so make doesn't depend on the editable-install .pth file
# (macOS can flag it hidden in iCloud-synced folders, and Python then skips it).
export PYTHONPATH := $(CURDIR)/src
EVALS := $(BIN)/python -m agent_evals
REPEAT ?= 5
TOLERANCE ?= 0.03

.PHONY: help install demo evals gate demo-regression baseline smoke test lint format clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-16s %s\n", $$1, $$2}'

$(INSTALLED): pyproject.toml
	@test -n "$(PYTHON)" || (echo "Python 3.10+ is required. Install it, or run: make PYTHON=/path/to/python3.12 demo" && exit 1)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -e ".[dev]"
	@touch $(INSTALLED)

install: $(INSTALLED) ## Create .venv and install the package with dev tools

demo: evals gate ## Run the suite (scripted model, no API key) and check it against the baseline

evals: $(INSTALLED) ## Run every case 5 times (REPEAT=n to change) into reports/latest.json
	$(EVALS) run --repeat $(REPEAT) --out reports/latest.json

gate: $(INSTALLED) ## Fail if reports/latest.json regressed versus baselines/scripted.json
	$(EVALS) compare baselines/scripted.json reports/latest.json --max-regression $(TOLERANCE)

demo-regression: $(INSTALLED) ## Simulate a bad prompt change and watch the gate catch it
	$(EVALS) run --model scripted:keep-it-short --repeat $(REPEAT) --out reports/regressed.json
	@echo
	@$(EVALS) compare baselines/scripted.json reports/regressed.json --max-regression $(TOLERANCE) \
		&& echo "Unexpected: the gate passed." \
		|| printf '\nThe gate failed, as it should: the change stopped the agent looking orders up.\n'


smoke: $(INSTALLED) ## Fast subset: cases tagged smoke, 3 runs each
	$(EVALS) run --tag smoke --repeat 3 --out reports/smoke.json

baseline: $(INSTALLED) ## Re-record baselines/scripted.json (do this on purpose, in its own commit)
	$(EVALS) run --repeat $(REPEAT) --out baselines/scripted.json

test: $(INSTALLED) ## Run the unit tests
	$(BIN)/pytest -q

lint: $(INSTALLED) ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: $(INSTALLED) ## Auto-format the code
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

clean: ## Remove the virtualenv and reports
	rm -rf $(VENV) reports .pytest_cache .ruff_cache src/*.egg-info
