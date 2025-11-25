# Makefile for ACO Multi-Objective Routing

.PHONY: help install test format type-check clean all

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
	conda env update -f ../conda-env.yaml --prune

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src/aco_routing --cov-report=html --cov-report=term

format:  ## Format code (black)
	black src/aco_routing tests/

format-check:  ## Check code formatting without changes
	black --check src/aco_routing tests/

type-check:  ## Run type checker (mypy)
	mypy src/aco_routing

all: format type-check test  ## Run all checks

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf results/*/

run:  ## Run experiment
	cd experiments && python run_experiment.py


