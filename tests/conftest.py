"""
Pytest configuration and fixtures for pyphysio tests.
"""

import pytest
import numpy as np


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--generate-wavelet-figures",
        action="store_true",
        default=False,
        help="Generate before/after filtering figures for WaveletFilter tests"
    )


@pytest.fixture(scope='session')
def rng():
    """Provide a seeded RNG for deterministic tests."""
    return np.random.RandomState(0)
