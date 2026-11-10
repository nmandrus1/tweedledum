# ------------------------------------------------------------------------------
# Part of Tweedledum Project.  This file is distributed under the MIT License.
# See accompanying file /LICENSE for details.
# ------------------------------------------------------------------------------
"""Shared pytest configuration.

Tests marked ``@pytest.mark.slow`` are skipped by default; run them with
``pytest --run-slow``.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run tests marked slow"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (enable with --run-slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: use --run-slow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
