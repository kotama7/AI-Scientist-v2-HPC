"""Worker process management for parallel execution.

This module provides worker process management for running tasks in parallel
with reliable termination on timeout.
"""

from ai_scientist.treesearch.worker.manager import (
    WorkerTask,
    WorkerResult,
    WorkerManager,
)

__all__ = [
    "WorkerTask",
    "WorkerResult",
    "WorkerManager",
]
