"""Dedicated database writer process for serializing SQLite writes.

This module provides a multiprocessing-based solution to SQLite's concurrent
write limitations. All write operations are serialized through a single
writer process, eliminating "database is locked" errors.

Architecture:
    Worker Process 1 ─┐
    Worker Process 2 ─┼─> Write Queue ─> DatabaseWriterProcess ─> SQLite DB
    Worker Process N ─┘

Usage:
    # Start writer process (in main process)
    writer = DatabaseWriterProcess(db_path)
    writer.start()

    # In worker processes, create MemoryManager with writer queue
    memory_manager = MemoryManager(db_path, config, writer_queue=writer.queue)

    # Shutdown when done
    writer.shutdown()
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Lazy singleton for multiprocessing.Manager (needed for picklable Queue)
_manager_instance: Optional[mp.managers.SyncManager] = None


def _get_manager() -> mp.managers.SyncManager:
    """Return a shared multiprocessing.Manager instance (created on first call)."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = mp.Manager()
    return _manager_instance


class WriteOpType(Enum):
    """Types of write operations."""
    EXECUTE = "execute"
    EXECUTE_MANY = "execute_many"
    COMMIT = "commit"
    EXECUTE_AND_COMMIT = "execute_and_commit"
    BATCH = "batch"  # Multiple operations in a single transaction
    SHUTDOWN = "shutdown"
    PING = "ping"  # Health check


@dataclass
class WriteRequest:
    """A database write request to be processed by the writer."""
    op_type: WriteOpType
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sql: Optional[str] = None
    params: Optional[Tuple] = None
    batch_ops: Optional[List[Tuple[str, Optional[Tuple]]]] = None  # For batch operations
    response_conn: Optional[Connection] = None  # Pipe connection for synchronous response
    timeout: float = 60.0  # Max wait time for response


@dataclass
class WriteResponse:
    """Response from the database writer."""
    request_id: str
    success: bool
    result: Any = None  # lastrowid, rowcount, or fetchall result
    error: Optional[str] = None
    error_type: Optional[str] = None


class DatabaseWriterProcess:
    """A dedicated process for handling all database writes.

    This class spawns a separate process that owns the database connection
    and processes write requests from a queue. This serializes all writes,
    eliminating concurrent write conflicts.

    Features:
    - Serialized writes eliminate "database is locked" errors
    - Batched operations for better performance
    - Automatic retry on transient errors
    - Graceful shutdown with queue draining
    - Health check via ping
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        max_batch_size: int = 100,
        batch_timeout: float = 0.05,  # Flush batch after 50ms of no new requests
        max_retries: int = 3,
    ):
        """Initialize the database writer.

        Args:
            db_path: Path to the SQLite database file.
            max_batch_size: Maximum operations to batch before flushing.
            batch_timeout: Seconds to wait for more operations before flushing.
            max_retries: Number of retries for transient errors.
        """
        self.db_path = Path(db_path)
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_retries = max_retries

        # Use Manager().Queue() instead of mp.Queue() to make the queue picklable
        # This allows passing the queue to worker processes via WorkerManager.submit()
        self._queue = _get_manager().Queue()
        self._process: Optional[mp.Process] = None
        self._started = mp.Event()
        self._shutdown_requested = mp.Event()
        self._shutdown_complete = mp.Event()

    @property
    def queue(self) -> Any:
        """The queue for submitting write requests.

        Returns a Manager-managed queue proxy that is picklable and can be
        passed to worker processes.
        """
        return self._queue

    def start(self) -> None:
        """Start the writer process."""
        if self._process is not None and self._process.is_alive():
            logger.warning("DatabaseWriterProcess already running")
            return

        self._started.clear()
        self._shutdown_requested.clear()
        self._shutdown_complete.clear()

        self._process = mp.Process(
            target=self._writer_loop,
            args=(
                self.db_path,
                self._queue,
                self._started,
                self._shutdown_requested,
                self._shutdown_complete,
                self.max_batch_size,
                self.batch_timeout,
                self.max_retries,
            ),
            daemon=False,  # Allow clean shutdown
            name="DatabaseWriterProcess",
        )
        self._process.start()

        # Wait for writer to be ready
        if not self._started.wait(timeout=30):
            raise RuntimeError("DatabaseWriterProcess failed to start within 30 seconds")

        logger.info("DatabaseWriterProcess started (pid=%d)", self._process.pid)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the writer process gracefully.

        Args:
            timeout: Maximum seconds to wait for shutdown.
        """
        if self._process is None or not self._process.is_alive():
            return

        logger.info("Shutting down DatabaseWriterProcess...")

        # Signal shutdown via event (reliable even if queue is broken)
        self._shutdown_requested.set()

        # Also send shutdown signal via queue for immediate pickup
        try:
            self._queue.put(WriteRequest(op_type=WriteOpType.SHUTDOWN), timeout=5)
        except Exception as e:
            logger.warning("Failed to send shutdown signal via queue: %s", e)

        # Wait for clean shutdown
        if not self._shutdown_complete.wait(timeout=timeout):
            logger.warning("Writer process did not shutdown cleanly, terminating...")
            self._process.terminate()
            self._process.join(timeout=30)

        if self._process.is_alive():
            logger.error("Writer process still alive after terminate, killing...")
            self._process.kill()
            self._process.join(timeout=10)

        logger.info("DatabaseWriterProcess shutdown complete")

    def is_alive(self) -> bool:
        """Check if the writer process is running."""
        return self._process is not None and self._process.is_alive()

    def ping(self, timeout: float = 5.0) -> bool:
        """Check if the writer process is responsive.

        Uses a simple liveness check since Queue objects cannot be pickled
        across process boundaries for response delivery.

        Args:
            timeout: Maximum seconds to wait for response (used for process check).

        Returns:
            True if writer process is alive and responsive, False otherwise.
        """
        if not self.is_alive():
            return False

        # Check if the process is still running and the queue is functional
        # We verify by checking the started event and process state
        if not self._started.is_set():
            return False

        # Additional check: verify process is responsive by checking exitcode
        if self._process.exitcode is not None:
            return False

        return True

    @staticmethod
    def _writer_loop(
        db_path: Path,
        queue: mp.Queue,
        started_event: mp.Event,
        shutdown_requested_event: mp.Event,
        shutdown_event: mp.Event,
        max_batch_size: int,
        batch_timeout: float,
        max_retries: int,
    ) -> None:
        """Main loop for the writer process (runs in separate process)."""
        conn: Optional[sqlite3.Connection] = None

        try:
            # Initialize database connection
            conn = sqlite3.connect(
                str(db_path),
                timeout=60,
                check_same_thread=True,  # Only this process writes
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")

            logger.info("DatabaseWriterProcess ready (db=%s)", db_path)
            started_event.set()

            pending_responses: List[Tuple[WriteRequest, Any]] = []

            while True:
                try:
                    # Check if shutdown was requested via event
                    if shutdown_requested_event.is_set():
                        logger.info("Writer detected shutdown event, draining queue...")
                        # Drain remaining items from the queue before exiting
                        while True:
                            try:
                                remaining = queue.get_nowait()
                                if remaining.op_type == WriteOpType.SHUTDOWN:
                                    break
                                result = DatabaseWriterProcess._execute_request(
                                    conn, remaining, max_retries
                                )
                                if remaining.response_conn:
                                    try:
                                        remaining.response_conn.send(result)
                                    except (BrokenPipeError, OSError):
                                        logger.debug("Response pipe closed for request %s (receiver gone)", remaining.request_id)
                                pending_responses.append((remaining, result))
                            except (Empty, OSError, EOFError):
                                break
                        if pending_responses:
                            DatabaseWriterProcess._flush_pending(
                                conn, pending_responses, max_retries
                            )
                        break

                    # Get next request with timeout
                    try:
                        request = queue.get(timeout=batch_timeout)
                    except (Empty, OSError, EOFError):
                        # Empty = normal timeout; OSError/EOFError = queue/manager gone
                        if pending_responses:
                            DatabaseWriterProcess._flush_pending(
                                conn, pending_responses, max_retries
                            )
                            pending_responses = []
                        continue

                    # Handle shutdown
                    if request.op_type == WriteOpType.SHUTDOWN:
                        logger.info("Writer received shutdown signal")
                        # Flush pending before shutdown
                        if pending_responses:
                            DatabaseWriterProcess._flush_pending(
                                conn, pending_responses, max_retries
                            )
                        break

                    # Handle ping (health check)
                    if request.op_type == WriteOpType.PING:
                        if request.response_conn:
                            try:
                                request.response_conn.send(WriteResponse(
                                    request_id=request.request_id,
                                    success=True,
                                ))
                            except (BrokenPipeError, OSError):
                                logger.debug("Response pipe closed for ping %s", request.request_id)
                        continue

                    # Process write operation
                    result = DatabaseWriterProcess._execute_request(
                        conn, request, max_retries
                    )

                    # Send response if requested (via Pipe connection)
                    if request.response_conn:
                        try:
                            request.response_conn.send(result)
                        except (BrokenPipeError, OSError):
                            logger.debug("Response pipe closed for request %s (receiver gone)", request.request_id)

                    # Batch commits for better performance
                    if request.op_type in (WriteOpType.EXECUTE, WriteOpType.EXECUTE_MANY):
                        pending_responses.append((request, result))
                        if len(pending_responses) >= max_batch_size:
                            DatabaseWriterProcess._flush_pending(
                                conn, pending_responses, max_retries
                            )
                            pending_responses = []

                except Exception as e:
                    logger.error("Error in writer loop: %s", e, exc_info=True)
                    # Try to recover
                    try:
                        conn.rollback()
                    except Exception:
                        pass

        except Exception as e:
            logger.error("Fatal error in DatabaseWriterProcess: %s", e, exc_info=True)
        finally:
            if conn:
                try:
                    conn.commit()  # Final commit
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.close()
                except Exception as e:
                    logger.error("Error closing database: %s", e)

            shutdown_event.set()
            logger.info("DatabaseWriterProcess exited")

    @staticmethod
    def _execute_request(
        conn: sqlite3.Connection,
        request: WriteRequest,
        max_retries: int,
    ) -> WriteResponse:
        """Execute a single write request with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                if request.op_type == WriteOpType.EXECUTE:
                    cursor = conn.execute(request.sql, request.params or ())
                    return WriteResponse(
                        request_id=request.request_id,
                        success=True,
                        result={"lastrowid": cursor.lastrowid, "rowcount": cursor.rowcount},
                    )

                elif request.op_type == WriteOpType.EXECUTE_MANY:
                    cursor = conn.executemany(request.sql, request.params or [])
                    return WriteResponse(
                        request_id=request.request_id,
                        success=True,
                        result={"rowcount": cursor.rowcount},
                    )

                elif request.op_type == WriteOpType.COMMIT:
                    conn.commit()
                    return WriteResponse(
                        request_id=request.request_id,
                        success=True,
                    )

                elif request.op_type == WriteOpType.EXECUTE_AND_COMMIT:
                    cursor = conn.execute(request.sql, request.params or ())
                    conn.commit()
                    return WriteResponse(
                        request_id=request.request_id,
                        success=True,
                        result={"lastrowid": cursor.lastrowid, "rowcount": cursor.rowcount},
                    )

                elif request.op_type == WriteOpType.BATCH:
                    results = []
                    for sql, params in (request.batch_ops or []):
                        cursor = conn.execute(sql, params or ())
                        results.append({
                            "lastrowid": cursor.lastrowid,
                            "rowcount": cursor.rowcount,
                        })
                    conn.commit()
                    return WriteResponse(
                        request_id=request.request_id,
                        success=True,
                        result=results,
                    )

                else:
                    return WriteResponse(
                        request_id=request.request_id,
                        success=False,
                        error=f"Unknown operation type: {request.op_type}",
                        error_type="ValueError",
                    )

            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                break
            except Exception as e:
                last_error = e
                break

        return WriteResponse(
            request_id=request.request_id,
            success=False,
            error=str(last_error),
            error_type=type(last_error).__name__ if last_error else None,
        )

    @staticmethod
    def _flush_pending(
        conn: sqlite3.Connection,
        pending: List[Tuple[WriteRequest, WriteResponse]],
        max_retries: int,
    ) -> None:
        """Commit pending operations."""
        if not pending:
            return

        for attempt in range(max_retries):
            try:
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                logger.error("Failed to flush pending commits: %s", e)
                raise


class DatabaseWriterClient:
    """Client for sending write requests to the DatabaseWriterProcess.

    This class provides a convenient interface for worker processes to
    send write requests to the centralized writer.

    Note: Synchronous operations use Pipe for cross-process response delivery.
    Pipe can be created in daemon processes (unlike Manager which cannot).
    """

    def __init__(
        self,
        queue: mp.Queue,
        default_timeout: float = 60.0,
        sync_by_default: bool = False,
    ):
        """Initialize the client.

        Args:
            queue: The writer process queue.
            default_timeout: Default timeout for synchronous operations.
            sync_by_default: If True, wait for confirmation on all writes.
        """
        self._queue = queue
        self._default_timeout = default_timeout
        self._sync_by_default = sync_by_default

    def _create_response_pipe(self) -> Tuple[Connection, Connection]:
        """Create a Pipe for cross-process response delivery.

        Returns a (recv_conn, send_conn) tuple. The send_conn is passed to the
        writer process via the request, and recv_conn is used to receive the response.
        Unlike Manager().Queue(), Pipe can be created in daemon processes.
        """
        return mp.Pipe(duplex=False)

    def execute(
        self,
        sql: str,
        params: Optional[Tuple] = None,
        sync: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> Optional[WriteResponse]:
        """Execute a SQL statement.

        Args:
            sql: SQL statement to execute.
            params: Parameters for the SQL statement.
            sync: If True, wait for confirmation. If None, use default.
            timeout: Timeout for synchronous operation.

        Returns:
            WriteResponse if sync=True, None otherwise.
        """
        sync = sync if sync is not None else self._sync_by_default
        timeout = timeout or self._default_timeout

        recv_conn, send_conn = self._create_response_pipe() if sync else (None, None)
        request = WriteRequest(
            op_type=WriteOpType.EXECUTE,
            sql=sql,
            params=params,
            response_conn=send_conn,
            timeout=timeout,
        )

        self._queue.put(request)

        if sync and recv_conn:
            try:
                if recv_conn.poll(timeout=timeout):
                    return recv_conn.recv()
                return WriteResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Timeout waiting for write confirmation",
                    error_type="TimeoutError",
                )
            finally:
                recv_conn.close()
                if send_conn:
                    send_conn.close()
        return None

    def execute_and_commit(
        self,
        sql: str,
        params: Optional[Tuple] = None,
        sync: bool = True,
        timeout: Optional[float] = None,
    ) -> Optional[WriteResponse]:
        """Execute a SQL statement and commit immediately.

        Args:
            sql: SQL statement to execute.
            params: Parameters for the SQL statement.
            sync: If True, wait for confirmation.
            timeout: Timeout for synchronous operation.

        Returns:
            WriteResponse if sync=True, None otherwise.
        """
        timeout = timeout or self._default_timeout

        recv_conn, send_conn = self._create_response_pipe() if sync else (None, None)
        request = WriteRequest(
            op_type=WriteOpType.EXECUTE_AND_COMMIT,
            sql=sql,
            params=params,
            response_conn=send_conn,
            timeout=timeout,
        )

        self._queue.put(request)

        if sync and recv_conn:
            try:
                if recv_conn.poll(timeout=timeout):
                    return recv_conn.recv()
                return WriteResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Timeout waiting for write confirmation",
                    error_type="TimeoutError",
                )
            finally:
                recv_conn.close()
                if send_conn:
                    send_conn.close()
        return None

    def commit(self, sync: bool = False, timeout: Optional[float] = None) -> Optional[WriteResponse]:
        """Request a commit.

        Args:
            sync: If True, wait for confirmation.
            timeout: Timeout for synchronous operation.

        Returns:
            WriteResponse if sync=True, None otherwise.
        """
        timeout = timeout or self._default_timeout

        recv_conn, send_conn = self._create_response_pipe() if sync else (None, None)
        request = WriteRequest(
            op_type=WriteOpType.COMMIT,
            response_conn=send_conn,
            timeout=timeout,
        )

        self._queue.put(request)

        if sync and recv_conn:
            try:
                if recv_conn.poll(timeout=timeout):
                    return recv_conn.recv()
                return WriteResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Timeout waiting for commit confirmation",
                    error_type="TimeoutError",
                )
            finally:
                recv_conn.close()
                if send_conn:
                    send_conn.close()
        return None

    def batch_execute(
        self,
        operations: List[Tuple[str, Optional[Tuple]]],
        sync: bool = True,
        timeout: Optional[float] = None,
    ) -> Optional[WriteResponse]:
        """Execute multiple operations in a single transaction.

        Args:
            operations: List of (sql, params) tuples.
            sync: If True, wait for confirmation.
            timeout: Timeout for synchronous operation.

        Returns:
            WriteResponse if sync=True, None otherwise.
        """
        timeout = timeout or self._default_timeout

        recv_conn, send_conn = self._create_response_pipe() if sync else (None, None)
        request = WriteRequest(
            op_type=WriteOpType.BATCH,
            batch_ops=operations,
            response_conn=send_conn,
            timeout=timeout,
        )

        self._queue.put(request)

        if sync and recv_conn:
            try:
                if recv_conn.poll(timeout=timeout):
                    return recv_conn.recv()
                return WriteResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Timeout waiting for batch confirmation",
                    error_type="TimeoutError",
                )
            finally:
                recv_conn.close()
                if send_conn:
                    send_conn.close()
        return None
