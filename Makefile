# Every target runs inside the uv environment. `make help` lists them.
PY := uv run python
# `make night`: length of the window in hours, and how long one run takes (measure it with the pilot)
HOURS ?= 7
T_RUN_S ?= 10800

.PHONY: help setup test test-all quality style data baseline experiments run night table

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /\t/'

setup: ## install the environment (uv.lock)
	uv sync

test: ## fast tests (no GPU, no network)
	uv run pytest

test-all: ## also trains tiny models on CPU and scores with EvalPlus (network, ~8 GB free RAM)
	uv run pytest -m ""

quality: ## check formatting and lint
	uv run ruff format --check . && uv run ruff check .

style: ## fix formatting and lint
	uv run ruff format . && uv run ruff check --fix .

data: ## build data/kodcode/ (once)
	$(PY) prepare_data.py

baseline: ## evaluate the untrained 1.5B and 0.5B models, with best-of-8
	$(PY) baseline.py --best-of-n 8
	$(PY) baseline.py --best-of-n 8 --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct

experiments: ## write experiments/ and queue.txt from methods.json
	$(PY) make_experiments.py

run: ## one experiment: make run EXP=experiments/lora/lr1e-5-tests
	$(PY) run.py $(EXP)

night: ## run queue.txt in order while the next run still fits in HOURS
	@deadline=$$(( $$(date +%s) + $(HOURS)*3600 - $(T_RUN_S) )); \
	while read -r exp; do \
	  case "$$exp" in ''|\#*) continue;; esac; \
	  res="results/$$(echo "$${exp#experiments/}" | sed 's#/#--#g').json"; \
	  [ -f "$$res" ] && continue; \
	  [ $$(date +%s) -le $$deadline ] || { echo "window closed before $$exp"; break; }; \
	  $(PY) run.py "$$exp" < /dev/null || echo "FAILED: $$exp"; \
	done < queue.txt

table: ## results/*.json → RESULTS.md
	$(PY) table.py
