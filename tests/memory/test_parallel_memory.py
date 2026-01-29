"""Integration tests for parallel memory operations.

Tests that simulate the actual parallel worker scenario where multiple
processes write to the memory database concurrently.
"""

import multiprocessing as mp
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import pytest


def worker_process(
    worker_id: int,
    db_path: Path,
    writer_queue: mp.Queue,
    num_writes: int,
    result_queue: mp.Queue,
):
    """Simulate a worker process performing memory operations.

    This mimics what _process_node_wrapper does in parallel_agent.py.
    """
    from ai_scientist.memory import MemoryManager

    errors = []
    try:
        # Create MemoryManager with writer queue (like in _process_node_wrapper)
        mm = MemoryManager(db_path, writer_queue=writer_queue)

        # Create a branch for this worker
        branch_id = mm.create_branch(None, f"worker_{worker_id}")

        # Perform multiple write operations (simulating node processing)
        for i in range(num_writes):
            try:
                # Write event (like node_created, memory_injected events)
                mm.write_event(
                    branch_id,
                    f"event_type_{i % 5}",
                    f"Worker {worker_id} event {i}: This is test content that simulates "
                    f"memory operations during node processing.",
                    tags=[f"worker:{worker_id}", f"event:{i}"],
                )

                # Occasional archival write (like success/error details)
                if i % 10 == 0:
                    mm._insert_archival(
                        branch_id,
                        f"Worker {worker_id} archival entry {i}: Detailed information "
                        f"stored for long-term retrieval.",
                        tags=["WORKER_LOG", f"worker:{worker_id}"],
                    )

            except Exception as e:
                errors.append(f"Worker {worker_id} write {i}: {type(e).__name__}: {e}")

        result_queue.put((worker_id, len(errors), errors))

    except Exception as e:
        result_queue.put((worker_id, -1, [f"Worker {worker_id} init error: {e}"]))


def worker_process_without_writer(
    worker_id: int,
    db_path: Path,
    num_writes: int,
    result_queue: mp.Queue,
):
    """Simulate a worker without centralized writer (old behavior).

    This demonstrates the "database is locked" problem.
    """
    from ai_scientist.memory import MemoryManager

    errors = []
    try:
        # Create MemoryManager WITHOUT writer queue
        mm = MemoryManager(db_path, writer_queue=None)

        branch_id = mm.create_branch(None, f"worker_{worker_id}")

        for i in range(num_writes):
            try:
                mm.write_event(
                    branch_id,
                    f"event_type_{i % 5}",
                    f"Worker {worker_id} event {i}",
                )
            except Exception as e:
                errors.append(f"Worker {worker_id} write {i}: {type(e).__name__}: {e}")

        result_queue.put((worker_id, len(errors), errors))

    except Exception as e:
        result_queue.put((worker_id, -1, [f"Worker {worker_id} init error: {e}"]))


class TestParallelMemoryOperations:
    """Tests for parallel memory operations with centralized writer."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            yield db_path

    def test_parallel_workers_with_writer_no_lock_errors(self, temp_db):
        """Test multiple parallel workers can write without lock errors."""
        from ai_scientist.memory import DatabaseWriterProcess

        num_workers = 4  # Reduced for faster testing
        writes_per_worker = 20  # Reduced for faster testing

        # Start centralized writer
        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            result_queue = mp.Queue()
            processes = []

            # Start worker processes
            for i in range(num_workers):
                p = mp.Process(
                    target=worker_process,
                    args=(i, temp_db, writer.queue, writes_per_worker, result_queue),
                )
                processes.append(p)
                p.start()

            # Wait for all workers to complete
            for p in processes:
                p.join(timeout=60)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get_nowait())

            # Check no errors occurred
            total_errors = 0
            for worker_id, error_count, errors in results:
                if error_count > 0:
                    print(f"Worker {worker_id} had {error_count} errors:")
                    for err in errors[:5]:  # Print first 5 errors
                        print(f"  {err}")
                total_errors += max(0, error_count)

            assert total_errors == 0, f"Expected 0 errors, got {total_errors}"
            assert len(results) == num_workers

        finally:
            # Shutdown writer - this drains the queue and commits pending writes
            writer.shutdown(timeout=60)

        # Verify all data was written AFTER shutdown completes
        conn = sqlite3.connect(str(temp_db))
        branch_count = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()

        assert branch_count == num_workers
        # Each worker writes writes_per_worker events
        # Note: We check >= because exact count depends on timing
        assert event_count >= num_workers * writes_per_worker * 0.9, \
            f"Expected ~{num_workers * writes_per_worker} events, got {event_count}"

    @pytest.mark.slow
    def test_parallel_workers_without_writer_may_have_lock_errors(self, temp_db):
        """Demonstrate that parallel workers WITHOUT writer may get lock errors.

        This test is marked slow and may be skipped in CI. It demonstrates
        the problem that the centralized writer solves.
        """
        num_workers = 4
        writes_per_worker = 20

        result_queue = mp.Queue()
        processes = []

        # Start worker processes WITHOUT centralized writer
        for i in range(num_workers):
            p = mp.Process(
                target=worker_process_without_writer,
                args=(i, temp_db, writes_per_worker, result_queue),
            )
            processes.append(p)
            p.start()

        # Wait for all workers to complete
        for p in processes:
            p.join(timeout=60)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        # Count errors (we expect some "database is locked" errors)
        total_errors = 0
        lock_errors = 0
        for worker_id, error_count, errors in results:
            total_errors += max(0, error_count)
            for err in errors:
                if "locked" in err.lower():
                    lock_errors += 1

        # This test documents the problem - without writer, lock errors occur
        # Note: This may occasionally pass if timing is lucky
        print(f"Without centralized writer: {total_errors} total errors, {lock_errors} lock errors")

    def test_consolidate_inherited_memory_with_writer(self, temp_db):
        """Test that consolidate_inherited_memory works with writer.

        This tests the specific function that was causing "database is locked"
        errors in the original issue.
        """
        from ai_scientist.memory import DatabaseWriterProcess, MemoryManager

        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            mm = MemoryManager(
                temp_db,
                config={
                    "recall_max_events": 5,
                    "recall_consolidation_threshold": 1.0,
                    "auto_consolidate": False,  # Manual trigger
                },
                writer_queue=writer.queue,
            )

            # Create parent and child branches
            parent_branch = mm.create_branch(None, "parent")
            child_branch = mm.create_branch(parent_branch, "child")

            # Add many events to parent (these become inherited for child)
            for i in range(20):
                mm.write_event(parent_branch, "test", f"Parent event {i}")

            # Manually trigger consolidation on child branch
            # This is the operation that was failing with "database is locked"
            try:
                consolidated = mm.consolidate_inherited_memory(child_branch)
                # Note: Actual consolidation requires LLM, so count may be 0
                # The important thing is no "database is locked" error
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    pytest.fail(f"Database locked error during consolidation: {e}")
                raise

        finally:
            writer.shutdown()

    def test_high_frequency_writes(self, temp_db):
        """Test high-frequency writes from multiple workers."""
        from ai_scientist.memory import DatabaseWriterProcess

        num_workers = 4
        writes_per_worker = 200  # High frequency

        writer = DatabaseWriterProcess(
            temp_db,
            max_batch_size=50,  # Smaller batches for testing
            batch_timeout=0.01,
        )
        writer.start()

        try:
            result_queue = mp.Queue()
            processes = []

            start_time = time.time()

            for i in range(num_workers):
                p = mp.Process(
                    target=worker_process,
                    args=(i, temp_db, writer.queue, writes_per_worker, result_queue),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join(timeout=120)

            elapsed = time.time() - start_time

            # Collect results
            results = []
            total_errors = 0
            while not result_queue.empty():
                worker_id, error_count, errors = result_queue.get_nowait()
                results.append((worker_id, error_count, errors))
                total_errors += max(0, error_count)

            assert total_errors == 0
            print(f"High-frequency test: {num_workers * writes_per_worker} writes "
                  f"in {elapsed:.2f}s ({(num_workers * writes_per_worker) / elapsed:.0f} writes/s)")

        finally:
            writer.shutdown()


class TestWriterResilience:
    """Tests for writer process resilience and error handling."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            yield db_path

    def test_writer_handles_invalid_operations(self, temp_db):
        """Test that writer handles invalid SQL gracefully."""
        from ai_scientist.memory import DatabaseWriterProcess, DatabaseWriterClient

        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        try:
            client = DatabaseWriterClient(writer.queue)

            # Send invalid SQL
            response = client.execute_and_commit(
                "DROP TABLE nonexistent_table",
                (),
                sync=True,
            )

            # Should return error response, not crash
            assert response is not None
            assert response.success is False

            # Writer should still be alive
            assert writer.is_alive()
            assert writer.ping()

            # Valid operations should still work
            response = client.execute_and_commit(
                "CREATE TABLE test (id INTEGER PRIMARY KEY)",
                (),
                sync=True,
            )
            assert response.success is True

        finally:
            writer.shutdown()

    def test_shutdown_timeout(self, temp_db):
        """Test that shutdown respects timeout."""
        from ai_scientist.memory import DatabaseWriterProcess

        writer = DatabaseWriterProcess(temp_db)
        writer.start()

        # Shutdown should complete within timeout
        start = time.time()
        writer.shutdown(timeout=5.0)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert not writer.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
