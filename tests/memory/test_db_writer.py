"""Unit tests for DatabaseWriterProcess.

Tests the centralized database writer that serializes SQLite writes
from multiple parallel processes to avoid "database is locked" errors.
"""

import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List
from unittest import TestCase

import pytest

from ai_scientist.memory.db_writer import (
    DatabaseWriterClient,
    DatabaseWriterProcess,
    WriteOpType,
    WriteRequest,
    WriteResponse,
)


class TestWriteRequest:
    """Tests for WriteRequest dataclass."""

    def test_create_request_with_defaults(self):
        """Test creating a request with default values."""
        request = WriteRequest(op_type=WriteOpType.EXECUTE)
        assert request.op_type == WriteOpType.EXECUTE
        assert request.request_id is not None
        assert len(request.request_id) == 32  # UUID hex
        assert request.sql is None
        assert request.params is None
        assert request.response_queue is None
        assert request.timeout == 60.0

    def test_create_request_with_sql(self):
        """Test creating a request with SQL and params."""
        request = WriteRequest(
            op_type=WriteOpType.EXECUTE,
            sql="INSERT INTO test (name) VALUES (?)",
            params=("value",),
        )
        assert request.sql == "INSERT INTO test (name) VALUES (?)"
        assert request.params == ("value",)

    def test_create_batch_request(self):
        """Test creating a batch request."""
        ops = [
            ("INSERT INTO test (name) VALUES (?)", ("a",)),
            ("INSERT INTO test (name) VALUES (?)", ("b",)),
        ]
        request = WriteRequest(
            op_type=WriteOpType.BATCH,
            batch_ops=ops,
        )
        assert request.op_type == WriteOpType.BATCH
        assert len(request.batch_ops) == 2


class TestWriteResponse:
    """Tests for WriteResponse dataclass."""

    def test_success_response(self):
        """Test creating a success response."""
        response = WriteResponse(
            request_id="abc123",
            success=True,
            result={"lastrowid": 1, "rowcount": 1},
        )
        assert response.success is True
        assert response.result["lastrowid"] == 1
        assert response.error is None

    def test_error_response(self):
        """Test creating an error response."""
        response = WriteResponse(
            request_id="abc123",
            success=False,
            error="database is locked",
            error_type="OperationalError",
        )
        assert response.success is False
        assert response.error == "database is locked"
        assert response.error_type == "OperationalError"


class TestDatabaseWriterProcess:
    """Tests for DatabaseWriterProcess."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            # Initialize the database with a test table
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()
            conn.close()
            yield db_path

    def test_start_and_shutdown(self, temp_db):
        """Test starting and shutting down the writer process."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()
        assert writer.is_alive()

        writer.shutdown()
        assert not writer.is_alive()

    def test_ping(self, temp_db):
        """Test health check via ping."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            assert writer.ping(timeout=5.0) is True
        finally:
            writer.shutdown()

    def test_ping_when_not_started(self, temp_db):
        """Test ping returns False when not started."""
        writer = DatabaseWriterProcess(temp_db)
        assert writer.ping() is False

    def test_execute_single_write(self, temp_db):
        """Test executing a single write operation."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            client = DatabaseWriterClient(writer.queue)
            response = client.execute_and_commit(
                "INSERT INTO test (name) VALUES (?)",
                ("test_value",),
                sync=True,
            )

            assert response is not None
            assert response.success is True
            assert response.result["rowcount"] == 1

            # Verify the write
            conn = sqlite3.connect(str(temp_db))
            row = conn.execute("SELECT name FROM test").fetchone()
            conn.close()
            assert row[0] == "test_value"
        finally:
            writer.shutdown()

    def test_execute_batch_operations(self, temp_db):
        """Test executing multiple operations in a batch."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            client = DatabaseWriterClient(writer.queue)
            operations = [
                ("INSERT INTO test (name) VALUES (?)", ("batch_1",)),
                ("INSERT INTO test (name) VALUES (?)", ("batch_2",)),
                ("INSERT INTO test (name) VALUES (?)", ("batch_3",)),
            ]
            response = client.batch_execute(operations, sync=True)

            assert response is not None
            assert response.success is True
            assert len(response.result) == 3

            # Verify all writes
            conn = sqlite3.connect(str(temp_db))
            rows = conn.execute("SELECT name FROM test ORDER BY id").fetchall()
            conn.close()
            assert [r[0] for r in rows] == ["batch_1", "batch_2", "batch_3"]
        finally:
            writer.shutdown()

    def test_async_write(self, temp_db):
        """Test async (non-blocking) write operation."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            client = DatabaseWriterClient(writer.queue, sync_by_default=False)
            # Async write should return None immediately
            response = client.execute(
                "INSERT INTO test (name) VALUES (?)",
                ("async_value",),
            )
            assert response is None

            # Commit and wait a bit for processing
            client.commit()
            time.sleep(0.5)

            # Verify the write eventually completes
            conn = sqlite3.connect(str(temp_db))
            row = conn.execute("SELECT name FROM test").fetchone()
            conn.close()
            assert row[0] == "async_value"
        finally:
            writer.shutdown()

    def test_concurrent_writes_from_multiple_clients(self, temp_db):
        """Test that multiple clients can write concurrently without locks."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            num_clients = 5
            writes_per_client = 20

            def worker_task(client_id: int, queue: mp.Queue):
                """Simulate a worker process writing to the database."""
                client = DatabaseWriterClient(queue, sync_by_default=False)
                for i in range(writes_per_client):
                    client.execute(
                        "INSERT INTO test (name) VALUES (?)",
                        (f"client_{client_id}_write_{i}",),
                    )
                client.commit(sync=True)

            # Start multiple processes writing concurrently
            processes = []
            for i in range(num_clients):
                p = mp.Process(target=worker_task, args=(i, writer.queue))
                processes.append(p)
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join(timeout=30)
                assert p.exitcode == 0

            # Verify all writes completed
            conn = sqlite3.connect(str(temp_db))
            count = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
            conn.close()
            assert count == num_clients * writes_per_client
        finally:
            writer.shutdown()

    def test_graceful_shutdown_drains_queue(self, temp_db):
        """Test that shutdown waits for pending writes to complete."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            client = DatabaseWriterClient(writer.queue, sync_by_default=False)

            # Queue many async writes
            for i in range(50):
                client.execute(
                    "INSERT INTO test (name) VALUES (?)",
                    (f"pending_{i}",),
                )
        finally:
            # Shutdown should drain the queue
            writer.shutdown(timeout=30)

        # Verify all writes were committed
        conn = sqlite3.connect(str(temp_db))
        count = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
        conn.close()
        assert count == 50

    def test_invalid_sql_returns_error(self, temp_db):
        """Test that invalid SQL returns an error response."""
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            client = DatabaseWriterClient(writer.queue)
            response = client.execute_and_commit(
                "INVALID SQL STATEMENT",
                (),
                sync=True,
            )

            assert response is not None
            assert response.success is False
            assert response.error is not None
            assert "syntax error" in response.error.lower()
        finally:
            writer.shutdown()


class TestDatabaseWriterClient:
    """Tests for DatabaseWriterClient."""

    def test_default_timeout(self):
        """Test client uses default timeout."""
        queue = mp.Queue()
        client = DatabaseWriterClient(queue, default_timeout=30.0)
        assert client._default_timeout == 30.0

    def test_sync_by_default_setting(self):
        """Test sync_by_default affects execute behavior."""
        queue = mp.Queue()

        # Sync by default
        client_sync = DatabaseWriterClient(queue, sync_by_default=True)
        assert client_sync._sync_by_default is True

        # Async by default
        client_async = DatabaseWriterClient(queue, sync_by_default=False)
        assert client_async._sync_by_default is False


class TestMemoryManagerWithWriter:
    """Integration tests for MemoryManager with centralized writer."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            yield db_path

    def test_memory_manager_uses_writer_queue(self, temp_db):
        """Test MemoryManager routes writes through writer queue."""
        from ai_scientist.memory import MemoryManager

        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            # Create MemoryManager with writer queue
            mm = MemoryManager(temp_db, writer_queue=writer.queue)
            assert mm._writer_client is not None

            # Create a branch (this should use the writer with sync=True)
            branch_id = mm.create_branch(None, "test_node")
            assert branch_id is not None

            # Write an event - this uses sync=False by default, so we need
            # to explicitly commit and wait
            mm.write_event(branch_id, "test_kind", "test event content")

            # Give time for async writes to complete (writer batches commits)
            time.sleep(0.5)

        finally:
            # Shutdown flushes all pending writes
            writer.shutdown()

        # Verify writes were persisted AFTER shutdown (which drains the queue)
        conn = sqlite3.connect(str(temp_db))
        branch_row = conn.execute(
            "SELECT id FROM branches WHERE id = ?", (branch_id,)
        ).fetchone()
        event_row = conn.execute(
            "SELECT text FROM events WHERE branch_id = ?", (branch_id,)
        ).fetchone()
        conn.close()

        assert branch_row is not None
        assert event_row is not None
        assert event_row[0] == "test event content"

    def test_memory_manager_without_writer_queue(self, temp_db):
        """Test MemoryManager works without writer queue (local writes)."""
        from ai_scientist.memory import MemoryManager

        # Create MemoryManager without writer queue
        mm = MemoryManager(temp_db, writer_queue=None)
        assert mm._writer_client is None

        # Operations should work with local connection
        branch_id = mm.create_branch(None, "test_node")
        mm.write_event(branch_id, "test_kind", "test event content")

        # Verify writes
        conn = sqlite3.connect(str(temp_db))
        event_row = conn.execute(
            "SELECT text FROM events WHERE branch_id = ?", (branch_id,)
        ).fetchone()
        conn.close()

        assert event_row is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
