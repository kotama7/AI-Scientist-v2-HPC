"""GPU management utilities.

This module provides GPU allocation and management for parallel
experiment execution.
"""

from typing import Dict, Set


def parse_cuda_visible_devices(value: str | None) -> list[str]:
    """Parse CUDA_VISIBLE_DEVICES environment variable.

    Args:
        value: Value of CUDA_VISIBLE_DEVICES (comma-separated GPU IDs).

    Returns:
        List of valid GPU ID strings.
    """
    if not value:
        return []
    tokens = [token.strip() for token in value.split(",")]
    return [token for token in tokens if token and token != "-1"]


class GPUManager:
    """Manages GPU allocation across processes.

    This class tracks which GPUs are available and assigns them to
    processes on request.

    Attributes:
        num_gpus: Total number of GPUs.
        available_gpus: List of available GPU IDs.
        available_gpu_set: Set of available GPU IDs for fast lookup.
        gpu_assignments: Mapping of process ID to assigned GPU ID.
    """

    def __init__(self, num_gpus: int, gpu_ids: list[str] | None = None):
        """Initialize GPU manager.

        Args:
            num_gpus: Number of GPUs to manage.
            gpu_ids: Optional list of specific GPU IDs to use.
        """
        self.num_gpus = num_gpus
        if gpu_ids:
            self.available_gpus = [str(gpu_id) for gpu_id in gpu_ids]
        else:
            self.available_gpus = [str(i) for i in range(num_gpus)]
        self.available_gpu_set: Set[str] = set(self.available_gpus)
        self.gpu_assignments: Dict[str, str] = {}  # process_id -> gpu_id

    def acquire_gpu(self, process_id: str) -> str:
        """Assigns a GPU to a process.

        Prefers to assign a GPU matching the process ID suffix if available.

        Args:
            process_id: Identifier for the requesting process.

        Returns:
            Assigned GPU ID string.

        Raises:
            RuntimeError: If no GPUs are available.
        """
        if not self.available_gpus:
            raise RuntimeError("No GPUs available")
        print(f"Available GPUs: {self.available_gpus}")
        print(f"Process ID: {process_id}")
        preferred_id = str(process_id).split("_")[-1]
        if preferred_id in self.available_gpu_set:
            gpu_id = preferred_id
        else:
            gpu_id = self.available_gpus[0]
        print(f"Acquiring GPU {gpu_id} for process {process_id}")
        self.available_gpus.remove(gpu_id)
        self.available_gpu_set.remove(gpu_id)
        self.gpu_assignments[process_id] = gpu_id
        print(f"GPU assignments: {self.gpu_assignments}")
        return gpu_id

    def release_gpu(self, process_id: str):
        """Releases GPU assigned to a process.

        Args:
            process_id: Identifier of the process releasing its GPU.
        """
        if process_id in self.gpu_assignments:
            gpu_id = self.gpu_assignments[process_id]
            if gpu_id not in self.available_gpu_set:
                self.available_gpus.append(gpu_id)
                self.available_gpu_set.add(gpu_id)
            del self.gpu_assignments[process_id]


# Backward compatibility alias
_parse_cuda_visible_devices = parse_cuda_visible_devices
