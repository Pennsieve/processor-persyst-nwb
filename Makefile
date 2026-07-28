.PHONY: help venv install test test-cov lint typecheck check pre-commit run clean

SERVICE_NAME     ?= "processor-persyst-nwb"
VENV_DIR         ?= venv
PYTHON           ?= python3

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make venv        - create virtual environment and install all dependencies"
	@echo "make install     - install dependencies into existing venv"
	@echo "make pre-commit  - install pre-commit hooks"
	@echo "make test        - run tests"
	@echo "make test-cov    - run tests with coverage report"
	@echo "make lint        - run linter + formatter with auto-fix"
	@echo "make typecheck   - run mypy --strict"
	@echo "make check       - run lint check, typecheck, and tests"
	@echo "make run         - build and run the processor via docker-compose"
	@echo "make clean       - remove all files from data/input and data/output"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r processor/requirements.txt
	$(VENV_DIR)/bin/pip install -r requirements-test.txt
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r processor/requirements.txt
	$(VENV_DIR)/bin/pip install -r requirements-test.txt

test:
	$(VENV_DIR)/bin/python -m pytest tests/ -v

test-cov:
	$(VENV_DIR)/bin/python -m pytest tests/ -v --cov=processor --cov-report=term-missing

lint:
	$(VENV_DIR)/bin/ruff check --fix processor/ tests/
	$(VENV_DIR)/bin/ruff format processor/ tests/

typecheck:
	$(VENV_DIR)/bin/mypy processor/

check:
	$(VENV_DIR)/bin/ruff check processor/ tests/
	$(VENV_DIR)/bin/ruff format --check processor/ tests/
	$(VENV_DIR)/bin/mypy processor/
	$(VENV_DIR)/bin/python -m pytest tests/

pre-commit:
	$(VENV_DIR)/bin/pre-commit install

run:
	docker-compose -f docker-compose.yml down --remove-orphans
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up --exit-code-from processor

clean:
	rm -rf data/input/*
	rm -rf data/output/*
