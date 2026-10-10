"""Pytest hooks for the Maidere suite."""

from maidere.core.db import stop_shared_connections


def pytest_sessionfinish(session, exitstatus) -> None:
    """Release pooled SQLite worker threads so pytest can exit."""
    stop_shared_connections()
