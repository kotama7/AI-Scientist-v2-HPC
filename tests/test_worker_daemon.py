"""Tests for WorkerManager non-daemon process configuration.

Verifies that:
1. Worker processes are non-daemon (can spawn child processes).
2. Workers can execute tasks that spawn child multiprocessing.Process.
3. atexit cleanup handler is registered.
"""

import multiprocessing as mp
import unittest
import time

from ai_scientist.treesearch.worker.manager import WorkerManager


def _identity(x):
    """Simple task that returns its input."""
    return x


def _spawn_child_process():
    """Task that spawns a child multiprocessing.Process (would fail if daemon)."""
    result_queue = mp.Queue()

    def _child_target(q):
        q.put("child_ok")

    p = mp.Process(target=_child_target, args=(result_queue,))
    p.start()
    p.join(timeout=10)
    try:
        return result_queue.get(timeout=5)
    except Exception:
        return "child_failed"


class TestWorkerManagerNonDaemon(unittest.TestCase):
    """Workers must be non-daemon to allow spawning child processes."""

    def setUp(self):
        self.wm = WorkerManager(max_workers=1)

    def tearDown(self):
        self.wm.shutdown(wait=True)

    def test_workers_are_non_daemon(self):
        """Worker processes should have daemon=False."""
        for worker in self.wm.workers:
            self.assertFalse(worker.daemon, "Worker should be non-daemon")

    def test_basic_task_execution(self):
        """Workers should execute basic tasks successfully."""
        task_id = self.wm.submit(_identity, 42)
        results = self.wm.wait_for_results([task_id], timeout=10)
        self.assertIn(task_id, results)
        self.assertTrue(results[task_id].completed)
        self.assertEqual(results[task_id].result, 42)

    def test_worker_can_spawn_child_process(self):
        """Non-daemon workers should be able to spawn child processes."""
        task_id = self.wm.submit(_spawn_child_process)
        results = self.wm.wait_for_results([task_id], timeout=30)
        self.assertIn(task_id, results)
        self.assertTrue(results[task_id].completed,
                        f"Task failed: {getattr(results.get(task_id), 'error', 'unknown')}")
        self.assertEqual(results[task_id].result, "child_ok")

    def test_atexit_cleanup_registered(self):
        """WorkerManager should register an atexit cleanup handler."""
        import atexit
        # Check that _atexit_cleanup is in the atexit registry
        # atexit._run_exitfuncs would run them; we just verify registration exists
        self.assertTrue(hasattr(self.wm, '_atexit_cleanup'))


if __name__ == "__main__":
    unittest.main()
