"""Worker process management for parallel task execution.

This module provides WorkerManager for running tasks in separate processes
with reliable termination on timeout.
"""

import atexit
import logging
import signal
import time
from multiprocessing import Process, Queue as MPQueue
from queue import Empty
from typing import Any, Callable, Dict, List

from rich import print

logger = logging.getLogger("ai-scientist")


class WorkerTask:
    """Represents a task to be executed by a worker process."""

    def __init__(self, task_id: int, func: Callable, args: tuple):
        self.task_id = task_id
        self.func = func
        self.args = args


class WorkerResult:
    """Represents the result from a worker process."""

    def __init__(self, task_id: int, result: Any = None, error: Exception = None):
        self.task_id = task_id
        self.result = result
        self.error = error
        self.completed = error is None


def _worker_process_target(task_queue: MPQueue, result_queue: MPQueue):
    """Target function for worker processes.

    Runs in a loop, pulling tasks from task_queue and putting results in result_queue.
    """
    # Ignore SIGINT in worker processes - let parent handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill - shutdown signal
                break

            try:
                result = task.func(*task.args)
                result_queue.put(WorkerResult(task.task_id, result=result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                result_queue.put(WorkerResult(task.task_id, error=e))
        except Empty:
            continue
        except Exception:
            # Queue was closed or other error
            break


class WorkerManager:
    """Manages worker processes using multiprocessing.Process for reliable termination.

    Unlike ProcessPoolExecutor, this class can reliably terminate worker processes
    on timeout using process.terminate().
    """

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.task_queue: MPQueue = MPQueue()
        self.result_queue: MPQueue = MPQueue()
        self.workers: List[Process] = []
        self.pending_tasks: Dict[int, WorkerTask] = {}
        self.next_task_id = 0
        self._shutdown = False
        self._start_workers()
        # Register atexit handler so non-daemon workers are cleaned up
        # if the main process exits unexpectedly.
        atexit.register(self._atexit_cleanup)

    def _start_workers(self):
        """Start worker processes."""
        for i in range(self.max_workers):
            p = Process(
                target=_worker_process_target,
                args=(self.task_queue, self.result_queue),
                name=f"Worker-{i}",
                daemon=False,  # Non-daemon so workers can spawn child processes (e.g. Interpreter)
            )
            try:
                p.start()
                self.workers.append(p)
                logger.info(f"Started worker process {p.name} (PID: {p.pid})")
            except Exception as e:
                logger.error(f"Failed to start worker process Worker-{i}: {e}")
                # Don't append failed process to workers list
                # Continue trying to start remaining workers

    def _atexit_cleanup(self):
        """Terminate workers on interpreter exit (safety net for non-daemon workers)."""
        if not self._shutdown:
            self.shutdown(wait=False)

    def submit(self, func: Callable, *args) -> int:
        """Submit a task and return a task_id.

        Returns:
            task_id: An integer identifying the submitted task
        """
        if self._shutdown:
            raise RuntimeError("WorkerManager has been shutdown")

        task_id = self.next_task_id
        self.next_task_id += 1
        task = WorkerTask(task_id, func, args)
        self.pending_tasks[task_id] = task
        self.task_queue.put(task)
        logger.info(f"Submitted task {task_id}")
        return task_id

    def wait_for_results(self, task_ids: List[int], timeout: float) -> Dict[int, WorkerResult]:
        """Wait for results from submitted tasks with timeout.

        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Dict mapping task_id to WorkerResult (for completed tasks)
        """
        results: Dict[int, WorkerResult] = {}
        remaining_ids = set(task_ids)
        start_time = time.time()

        while remaining_ids and (time.time() - start_time) < timeout:
            try:
                # Poll for results with short timeout
                result = self.result_queue.get(timeout=1.0)
                if result.task_id in remaining_ids:
                    results[result.task_id] = result
                    remaining_ids.remove(result.task_id)
                    if result.task_id in self.pending_tasks:
                        del self.pending_tasks[result.task_id]
                    logger.info(f"Received result for task {result.task_id}")
            except Empty:
                continue

        # Log timeout information
        if remaining_ids:
            elapsed = time.time() - start_time
            logger.warning(f"Timeout after {elapsed:.1f}s. Tasks not completed: {remaining_ids}")

        return results

    def terminate_and_restart(self):
        """Terminate all workers and restart fresh ones.

        This is called after a timeout to ensure no zombie processes remain.
        """
        logger.warning("Terminating all workers and restarting...")
        print("[yellow]Terminating timed-out workers and restarting...[/yellow]")

        # Terminate all existing workers
        for worker in self.workers:
            # Check if process was actually started (has valid _popen handle)
            if getattr(worker, '_popen', None) is None:
                logger.warning(f"Worker {worker.name} was never started, skipping termination")
                continue
            if worker.is_alive():
                logger.info(f"Terminating worker {worker.name} (PID: {worker.pid})")
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not terminate, killing...")
                    worker.kill()
                    worker.join(timeout=2)

        # Clear queues
        self._drain_queue(self.task_queue)
        self._drain_queue(self.result_queue)

        # Clear pending tasks
        self.pending_tasks.clear()

        # Create new queues (old ones may be corrupted after termination)
        self.task_queue = MPQueue()
        self.result_queue = MPQueue()

        # Start fresh workers
        self.workers = []
        self._start_workers()

        logger.info("Worker restart complete")
        print("[green]Workers restarted successfully[/green]")

    def _drain_queue(self, q: MPQueue):
        """Drain all items from a queue."""
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass
        except Exception:
            pass

    def shutdown(self, wait: bool = True):
        """Shutdown all workers.

        Args:
            wait: If True, wait for workers to finish gracefully before terminating
        """
        if self._shutdown:
            return

        self._shutdown = True
        logger.info("Shutting down WorkerManager...")

        # Send poison pills to workers
        for _ in self.workers:
            try:
                self.task_queue.put(None)
            except Exception:
                pass

        if wait:
            # Wait briefly for graceful shutdown
            for worker in self.workers:
                worker.join(timeout=2)

        # Force terminate any remaining workers
        for worker in self.workers:
            # Check if process was actually started (has valid _popen handle)
            if getattr(worker, '_popen', None) is None:
                continue
            if worker.is_alive():
                logger.info(f"Force terminating worker {worker.name}")
                worker.terminate()
                worker.join(timeout=2)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=1)

        self.workers = []
        logger.info("WorkerManager shutdown complete")
