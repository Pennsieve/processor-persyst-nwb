.PHONY: help venv install lock test test-cov lint typecheck check pre-commit run clean

SERVICE_NAME     ?= "processor-persyst-nwb"
VENV_DIR         ?= venv
PYTHON           ?= python3

# The digest-pinned base image in the Dockerfile is the only source of this
# value. The lock files therefore use the same interpreter as the container.
BASE_IMAGE       := $(shell sed -n 's/^FROM //p' Dockerfile)
LOCK_PLATFORM    ?= linux/amd64

# The container runs as the user who starts it, so the bind-mounted output stays
# writable on the host. Do not rename these to UID and GID: sh makes those two
# variables read-only, the assignment fails, and the container then uses the UID
# of the image.
COMPOSE          := HOST_UID=$(shell id -u) HOST_GID=$(shell id -g) \
                    docker-compose -f docker-compose.yml

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make venv        - create virtual environment and install all dependencies"
	@echo "make install     - install dependencies into existing venv"
	@echo "make lock        - regenerate the hash-pinned dependency locks"
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
	$(VENV_DIR)/bin/pip install --require-hashes -r requirements-test.lock
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install --require-hashes -r requirements-test.lock

# This target resolves the versions in the base image of the container, so the
# pins match production rather than the local machine. It compiles
# requirements-test.txt together with the runtime file, so the two lock files
# agree on each shared transitive version.
lock:
	docker run --rm --platform $(LOCK_PLATFORM) \
		-v "$(CURDIR)":/w -w /w $(BASE_IMAGE) sh -c '\
		pip install --quiet --no-cache-dir pip-tools && \
		pip-compile --quiet --generate-hashes --allow-unsafe --strip-extras \
			--output-file processor/requirements.lock \
			processor/requirements.txt && \
		pip-compile --quiet --generate-hashes --allow-unsafe --strip-extras \
			--output-file requirements-test.lock \
			processor/requirements.txt requirements-test.txt'

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
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) build
	$(COMPOSE) up --exit-code-from processor

clean:
	rm -rf data/input/*
	rm -rf data/output/*
