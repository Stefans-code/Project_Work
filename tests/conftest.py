"""
Pytest configuration and fixtures for pyphysio tests.
"""

import pytest
import numpy as np
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--generate-figures",
        action="store_true",
        default=False,
        help="Generate diagnostic figures for generator and filter tests (before/after)"
    )


@pytest.fixture(scope='session')
def rng():
    """Provide a seeded RNG for deterministic tests."""
    return np.random.RandomState(0)


@pytest.fixture(scope='session', autouse=True)
def _seed_numpy_global():
    """Seed numpy's global RNG for tests that still call `np.random.*`.

    This is a conservative fallback to ensure determinism for tests
    that haven't yet been migrated to use the `rng` fixture.
    """
    np.random.seed(0)
    yield


@pytest.fixture
def generate_figures(request):
    """Global fixture to check whether figures should be generated."""
    return request.config.getoption("--generate-figures")


@pytest.fixture
def figure_dir(request, generate_figures):
    """Create a per-module figure directory when `--generate-figures` is set.

    Directory will be `<test_module>_figures` next to the test file.
    """
    if not generate_figures:
        return None
    test_path = Path(request.node.fspath)
    fig_dir = test_path.parent / f"{test_path.stem}_figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir
