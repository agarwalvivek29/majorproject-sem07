# =============================================================================
# Makefile - Deepfake Detection System
# =============================================================================
# Common commands for development, testing, and deployment
#
# Usage:
#   make install      Install dependencies
#   make test         Run tests
#   make lint         Run linters
#   make docker       Build Docker image
#   make help         Show all commands
# =============================================================================

.PHONY: help install install-dev test lint format clean docker train detect realtime

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
IMAGE_NAME := deepfake-detector
IMAGE_TAG := latest

# =============================================================================
# Help
# =============================================================================
help:
	@echo "Deepfake Detection System - Available Commands"
	@echo "==============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make setup-pre     Setup pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run all linters"
	@echo "  make format        Format code with black and isort"
	@echo "  make typecheck     Run mypy type checker"
	@echo ""
	@echo "Application:"
	@echo "  make train         Train a model (DATA_DIR=path/to/data)"
	@echo "  make detect        Detect deepfake (VIDEO=path/to/video)"
	@echo "  make realtime      Start real-time detection (SOURCE=url)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker        Build Docker image"
	@echo "  make docker-dev    Build development image"
	@echo "  make docker-gpu    Build GPU image"
	@echo "  make docker-run    Run detector container"
	@echo "  make compose-up    Start all services"
	@echo "  make compose-down  Stop all services"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make clean-all     Remove all generated files"
	@echo "  make docs          Build documentation"
	@echo ""

# =============================================================================
# Setup
# =============================================================================
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

setup-pre:
	pre-commit install
	pre-commit autoupdate

# =============================================================================
# Testing
# =============================================================================
test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-integration:
	$(PYTHON) -m pytest tests/ -v -m "integration"

# =============================================================================
# Code Quality
# =============================================================================
lint: lint-black lint-isort lint-flake8 lint-pylint

lint-black:
	$(PYTHON) -m black --check src/ scripts/ tests/

lint-isort:
	$(PYTHON) -m isort --check-only src/ scripts/ tests/

lint-flake8:
	$(PYTHON) -m flake8 src/ scripts/ tests/

lint-pylint:
	$(PYTHON) -m pylint src/ --exit-zero

format:
	$(PYTHON) -m black src/ scripts/ tests/
	$(PYTHON) -m isort src/ scripts/ tests/

typecheck:
	$(PYTHON) -m mypy src/

# =============================================================================
# Application Commands
# =============================================================================
DATA_DIR ?= data/raw
MODEL_DIR ?= models
VIDEO ?= video.mp4
SOURCE ?= 0

train:
	$(PYTHON) scripts/train.py --dataset $(DATA_DIR) --output $(MODEL_DIR)

detect:
	$(PYTHON) scripts/detect.py $(VIDEO) --model $(MODEL_DIR)

realtime:
	$(PYTHON) scripts/realtime.py $(SOURCE) --model $(MODEL_DIR)

# =============================================================================
# Docker
# =============================================================================
docker:
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-dev:
	$(DOCKER) build --target dev -t $(IMAGE_NAME):dev .

docker-gpu:
	$(DOCKER) build --target gpu -t $(IMAGE_NAME):gpu .

docker-run:
	$(DOCKER) run -it --rm \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/models:/app/models:ro \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		scripts/detect.py --help

docker-shell:
	$(DOCKER) run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		/bin/bash

compose-up:
	$(DOCKER_COMPOSE) up -d

compose-down:
	$(DOCKER_COMPOSE) down

compose-logs:
	$(DOCKER_COMPOSE) logs -f

compose-build:
	$(DOCKER_COMPOSE) build

# =============================================================================
# Documentation
# =============================================================================
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# =============================================================================
# Cleanup
# =============================================================================
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

clean-all: clean
	rm -rf logs/
	rm -rf output/
	rm -rf .venv/
	rm -rf venv/

clean-docker:
	$(DOCKER) rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	$(DOCKER) rmi $(IMAGE_NAME):dev || true
	$(DOCKER) rmi $(IMAGE_NAME):gpu || true

# =============================================================================
# Development Utilities
# =============================================================================
run-jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

profile:
	$(PYTHON) -m cProfile -o profile.stats scripts/detect.py $(VIDEO)
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

benchmark:
	$(PYTHON) -m pytest tests/benchmarks/ -v --benchmark-only
