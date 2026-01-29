"""Test that database writer queue is picklable for multiprocessing.

This test verifies that the queue used by DatabaseWriterProcess can be
passed to worker processes via pickle (required by WorkerManager.submit).
"""

import pickle
import tempfile
import unittest
from pathlib import Path


class TestQueuePickling(unittest.TestCase):
    """Test that Manager-based queue is picklable."""

    def test_database_writer_queue_is_picklable(self):
        """Test that DatabaseWriterProcess queue can be pickled."""
        from ai_scientist.memory.db_writer import DatabaseWriterProcess

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            writer = DatabaseWriterProcess(db_path)
            queue = writer.queue

            # This should not raise an error
            try:
                pickled = pickle.dumps(queue)
                unpickled = pickle.loads(pickled)
                self.assertIsNotNone(unpickled)
            except Exception as e:
                self.fail(
                    f"Queue should be picklable but got: {e}\n"
                    "Use multiprocessing.Manager().Queue() instead of mp.Queue()"
                )

    def test_queue_can_be_passed_as_argument(self):
        """Test that queue can be serialized as part of function arguments."""
        from ai_scientist.memory.db_writer import DatabaseWriterProcess

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            writer = DatabaseWriterProcess(db_path)
            queue = writer.queue

            # Simulate passing queue as part of args tuple
            # (this is what WorkerManager.submit does with the arguments)
            args = (queue, "other_arg", 123)

            try:
                pickled = pickle.dumps(args)
                unpickled_args = pickle.loads(pickled)
                self.assertEqual(len(unpickled_args), 3)
                self.assertIsNotNone(unpickled_args[0])  # queue
                self.assertEqual(unpickled_args[1], "other_arg")
                self.assertEqual(unpickled_args[2], 123)
            except Exception as e:
                self.fail(
                    f"Args tuple with queue should be picklable but got: {e}"
                )


if __name__ == "__main__":
    unittest.main()
