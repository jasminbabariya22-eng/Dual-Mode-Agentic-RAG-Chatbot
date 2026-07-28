"""
Pytest configuration for the test suite.

Sets asyncio_mode to 'auto' so all async test functions are automatically
run under an event loop without needing per-test @pytest.mark.asyncio decorators.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
