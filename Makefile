# Convenience commands for working on tweedledum.  Run `make help` for a list.
#
# Everything here works inside whatever Python environment is active (a venv,
# uv, or conda).  Override variables on the command line, e.g.
#     make test-cpp JOBS=4 CXX=clang++

PYTHON   ?= python3
BUILD    ?= build/cpp
JOBS     ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
CMAKE_ARGS ?=
# Use uv when it is installed (uv-created venvs do not contain pip).
PIP_INSTALL ?= $(shell command -v uv >/dev/null 2>&1 \
                 && echo "uv pip install --python $(shell command -v $(PYTHON))" \
                 || echo "$(PYTHON) -m pip install")

.PHONY: help install dev test test-python test-python-all test-cpp build-cpp clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Build and install the Python package (non-editable)
	$(PIP_INSTALL) .

dev:  ## Editable install with test dependencies (re-run after changing C++ code)
	$(PIP_INSTALL) -e ".[test]"

test: test-cpp test-python  ## Run the C++ and the Python test suites

test-python:  ## Run the Python tests (requires `make dev` first)
	$(PYTHON) -m pytest

test-python-all:  ## Run the Python tests, including slow ones
	$(PYTHON) -m pytest --run-slow

build-cpp:  ## Configure and build the C++ library and tests
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DTWEEDLEDUM_TESTS=ON $(CMAKE_ARGS)
	cmake --build $(BUILD) -j $(JOBS)

test-cpp: build-cpp  ## Build and run the C++ tests
	ctest --test-dir $(BUILD) --output-on-failure

clean:  ## Remove build artifacts
	rm -rf build dist *.egg-info .pytest_cache
	find python -name '__pycache__' -type d -prune -exec rm -rf {} +
