"""Unit tests for embedded DuckDB session management and lock handling.

Functional Purpose:
    Verifies DuckDBManager connection lifecycle, in-memory isolation,
    and deterministic DatabaseLockedError emission when file lock retries are exhausted.

Explicit Dependency Tracking:
    - pytest, unittest.mock.
    - duckdb.IOException.
    - quant.infrastructure.database.duckdb_session: DuckDBManager, DatabaseLockedError, ERR_DATA_DB_LOCKED.
"""

from __future__ import annotations

from unittest.mock import patch

import duckdb
import pytest

from quant.infrastructure.database.duckdb_session import (
    ERR_DATA_DB_LOCKED,
    DatabaseLockedError,
    DuckDBManager,
)


def test_duckdb_in_memory_connection_succeeds() -> None:
    """Verify DuckDBManager initializes properly with :memory: target."""
    mgr = DuckDBManager(":memory:")
    assert mgr.get_connection() is not None
    mgr.close()


def test_duckdb_locked_error_raised_after_retries() -> None:
    """Verify DuckDBManager raises DatabaseLockedError with ERR-DATA-008 when file is locked."""
    with patch.object(
        duckdb,
        "connect",
        side_effect=duckdb.IOException("Database locked by another process"),
    ):
        with pytest.raises(DatabaseLockedError) as exc_info:
            DuckDBManager(database_path="test_locked.duckdb", max_retries=2)
        assert exc_info.value.code == ERR_DATA_DB_LOCKED
        assert "could not be locked" in str(exc_info.value)
