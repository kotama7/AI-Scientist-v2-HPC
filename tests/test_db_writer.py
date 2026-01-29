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


if __name__ == "__main__":
    unittest.main()
