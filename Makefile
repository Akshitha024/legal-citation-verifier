.PHONY: help install lint typecheck test verify report plots clean

DATA ?= tests/fixtures/cases.jsonl
PROVIDER ?= heuristic

help:
	@echo "make install                       - install deps via uv"
	@echo "make lint / typecheck / test       - quality gates"
	@echo "make verify DATA=path              - run the verifier loop on a JSONL"
	@echo "make report                        - aggregate per-run summary"
	@echo "make plots                         - regenerate the 5 chart types"

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not slow and not needs_provider"

verify:
	uv run lcg verify --data $(DATA) --provider $(PROVIDER)

report:
	uv run lcg report

plots:
	uv run lcg plots --out-dir results/figures

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


.PHONY: pdf test-artifacts
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf --pdf-engine=xelatex --toc --toc-depth=2 --number-sections -V geometry:margin=1in -V fontsize=11pt -V mainfont="Helvetica" -V monofont="Menlo" -V linkcolor=blue -V urlcolor=blue -V linestretch=1.15 || echo "pandoc + xelatex required; see https://pandoc.org/installing.html"

test-artifacts:
	uv run python ../../_meta/retrofit.py "$(notdir $(CURDIR))" "$(notdir $(CURDIR))"
