"""
Utilities for generating final memory for paper writeup.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_node_data(node) -> Dict[str, Any]:
    """Extract comprehensive data from a node for paper generation."""
    return {
        "id": node.id,
        "branch_id": node.branch_id,
        "plan": node.plan or "",
        "overall_plan": node.overall_plan or "",
        "code": node.code or "",
        "phase_artifacts": node.phase_artifacts or {},
        "analysis": node.analysis or "",
        "metric": {
            "value": node.metric.value if node.metric else None,
            "name": getattr(node.metric, "name", "") if node.metric else "",
        },
        "exp_results_dir": node.exp_results_dir or "",
        "plot_analyses": node.plot_analyses or [],
        "vlm_feedback_summary": node.vlm_feedback_summary or [],
        "datasets_successfully_tested": node.datasets_successfully_tested or [],
        "plot_paths": node.plot_paths or [],
        "exec_time_feedback": node.exec_time_feedback or "",
        "workspace_path": node.workspace_path or "",
    }


def extract_top_node_data(node) -> Dict[str, Any]:
    """Extract minimal data from a top node for comparative analysis."""
    return {
        "id": node.id,
        "branch_id": node.branch_id,
        "plan": node.plan or "",
        "metric": {
            "value": node.metric.value if node.metric else None,
            "name": getattr(node.metric, "name", "") if node.metric else "",
        },
        "analysis": node.analysis or "",
        "vlm_feedback_summary": node.vlm_feedback_summary or [],
    }


def find_best_node(manager):
    """Find the best node (highest metric) from all journals."""
    best_node = None
    for journal in manager.journals.values():
        for node in journal.nodes:
            if node.is_buggy:
                continue
            if node.metric is None:
                continue
            if best_node is None or node.metric > best_node.metric:
                best_node = node
    return best_node


def collect_top_nodes(manager, top_n: int = 5):
    """Collect top N nodes by metric from all journals."""
    top_nodes = []
    for journal in manager.journals.values():
        for node in journal.nodes:
            if node.is_buggy:
                continue
            if node.metric is None:
                continue
            top_nodes.append(node)
    # Sort by metric (descending) and take top N
    top_nodes.sort(key=lambda n: n.metric, reverse=True)
    return top_nodes[:top_n]


def generate_paper_memory_from_manager(
    memory_manager,
    manager,
    workspace_dir: Path,
    log_dir: Path,
    root_branch_id: str,
):
    """
    Generate final memory for paper writeup from AgentManager state.

    Args:
        memory_manager: MemoryManager instance
        manager: AgentManager instance with experiment results
        workspace_dir: Path to workspace directory
        log_dir: Path to log directory
        root_branch_id: Root branch ID for memory system
    """
    best_node = find_best_node(manager)
    top_nodes = collect_top_nodes(manager, top_n=5)

    # Extract comprehensive node data for paper generation
    best_node_data = None
    if best_node:
        best_node_data = extract_node_data(best_node)

    # Extract data from top N nodes for comparative analysis
    top_nodes_data = [extract_top_node_data(node) for node in top_nodes]

    artifacts_index = {
        "log_dir": str(log_dir),
        "workspace_dir": str(workspace_dir),
        "best_node_id": getattr(best_node, "id", None),
        "best_node_data": best_node_data,
        "top_nodes_data": top_nodes_data,
    }

    try:
        memory_manager.generate_final_memory_for_paper(
            run_dir=workspace_dir,
            root_branch_id=root_branch_id or "",
            best_branch_id=getattr(best_node, "branch_id", None),
            artifacts_index=artifacts_index,
        )
    except Exception as exc:
        logger.warning("Failed to generate final memory: %s", exc)
        raise
