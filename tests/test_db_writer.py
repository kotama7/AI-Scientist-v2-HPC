"""Tests for DatabaseWriterProcess and _get_manager.

Verifies that:
1. _get_manager() returns a valid multiprocessing.Manager instance.
2. _get_manager() returns the same singleton on repeated calls.
3. DatabaseWriterProcess can be instantiated (queue creation via _get_manager works).
"""

import multiprocessing as mp
import unittest

from ai_scientist.memory.db_writer import _get_manager, DatabaseWriterProcess


class TestGetManager(unittest.TestCase):
    """Tests for the _get_manager singleton."""

    def test_returns_sync_manager(self):
        """_get_manager() should return a SyncManager instance."""
        mgr = _get_manager()
        self.assertIsInstance(mgr, mp.managers.SyncManager)

    def test_singleton(self):
        """Repeated calls should return the same instance."""
        mgr1 = _get_manager()
        mgr2 = _get_manager()
        self.assertIs(mgr1, mgr2)

    def test_queue_creation(self):
        """Manager should be able to create a picklable Queue."""
        mgr = _get_manager()
        q = mgr.Queue()
        # Verify it works
        q.put("test")
        self.assertEqual(q.get(timeout=2), "test")


class TestDatabaseWriterProcessInit(unittest.TestCase):
    """Tests for DatabaseWriterProcess instantiation."""

    def test_instantiation_creates_queue(self):
        """DatabaseWriterProcess.__init__ should succeed (no NameError on _get_manager)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            writer = DatabaseWriterProcess(db_path)
            # Queue should be created successfully
            self.assertIsNotNone(writer.queue)


class TestBrokenPipeHandling(unittest.TestCase):
    """Verify that _writer_loop handles BrokenPipeError on response_conn.send().

    Regression test: when the requesting process exits before the writer sends
    its response, the pipe's read end is closed, causing BrokenPipeError.
    The writer must not crash in this scenario.
    """

    def test_writer_survives_broken_response_pipe(self):
        """Writer loop must not crash when response pipe is closed before send."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_bp.sqlite"
            writer = DatabaseWriterProcess(db_path)
            writer.start()
            try:
                # Create a pipe, then close the recv end to simulate the
                # requesting process going away.
                recv_conn, send_conn = mp.Pipe(duplex=False)
                recv_conn.close()  # Simulate requester gone

                from ai_scientist.memory.db_writer import WriteRequest, WriteOpType
                request = WriteRequest(
                    op_type=WriteOpType.EXECUTE_AND_COMMIT,
                    sql="CREATE TABLE IF NOT EXISTS bp_test (id INTEGER PRIMARY KEY)",
                    response_conn=send_conn,
                )
                writer.queue.put(request)

                # Give the writer time to process
                import time
                time.sleep(1)

                # Writer should still be alive
                self.assertTrue(writer.is_alive(), "Writer process crashed on BrokenPipeError")

                # Verify the writer is functional by doing another write
                recv2, send2 = mp.Pipe(duplex=False)
                request2 = WriteRequest(
                    op_type=WriteOpType.EXECUTE_AND_COMMIT,
                    sql="INSERT INTO bp_test (id) VALUES (1)",
                    response_conn=send2,
                )
                writer.queue.put(request2)
                if recv2.poll(timeout=5):
                    resp = recv2.recv()
                    self.assertTrue(resp.success, f"Follow-up write failed: {resp.error}")
                else:
                    self.fail("Writer did not respond after BrokenPipeError recovery")
                recv2.close()
                send2.close()
            finally:
                writer.shutdown(timeout=10)

    def test_writer_survives_broken_pipe_during_shutdown_drain(self):
        """Writer must handle BrokenPipeError when draining queue during shutdown."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_bp_drain.sqlite"
            writer = DatabaseWriterProcess(db_path)
            writer.start()
            try:
                from ai_scientist.memory.db_writer import WriteRequest, WriteOpType
                # Ensure table exists
                recv_setup, send_setup = mp.Pipe(duplex=False)
                writer.queue.put(WriteRequest(
                    op_type=WriteOpType.EXECUTE_AND_COMMIT,
                    sql="CREATE TABLE IF NOT EXISTS drain_test (id INTEGER PRIMARY KEY)",
                    response_conn=send_setup,
                ))
                recv_setup.poll(timeout=5)
                recv_setup.recv()
                recv_setup.close()
                send_setup.close()

                # Queue a request with a broken pipe, then immediately shutdown
                recv_conn, send_conn = mp.Pipe(duplex=False)
                recv_conn.close()  # Break the pipe
                writer.queue.put(WriteRequest(
                    op_type=WriteOpType.EXECUTE_AND_COMMIT,
                    sql="INSERT INTO drain_test (id) VALUES (42)",
                    response_conn=send_conn,
                ))
            finally:
                # Shutdown should complete without hanging or crashing
                writer.shutdown(timeout=10)


if __name__ == "__main__":
    unittest.main()
