"""TreeSearch module for AI Scientist.

This module provides the core tree search functionality including:
- Worker process management for parallel execution
- Agent classes for experiment orchestration
- GPU resource management
- Ablation experiment tracking
- Journal and node management
"""

# Re-export main classes and functions
from ai_scientist.treesearch.worker import (
    WorkerTask,
    WorkerResult,
    WorkerManager,
)
from ai_scientist.treesearch.gpu import (
    GPUManager,
    parse_cuda_visible_devices,
)
from ai_scientist.treesearch.ablation import (
    AblationConfig,
    AblationIdea,
    HyperparamTuningIdea,
)
from ai_scientist.treesearch.journal import (
    Journal,
    Node,
)

__all__ = [
    # Worker
    "WorkerTask",
    "WorkerResult",
    "WorkerManager",
    # GPU
    "GPUManager",
    "parse_cuda_visible_devices",
    # Ablation
    "AblationConfig",
    "AblationIdea",
    "HyperparamTuningIdea",
    # Journal
    "Journal",
    "Node",
]
