"""Export journal to HTML visualization of tree + code."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

from rich import print

from ..journal import Journal

# Import from refactored modules
from .tree_layout import get_edges, generate_layout, normalize_layout
from .stage_helpers import get_completed_stages, get_stage_dir_map, stage_dir_to_stage_id
from .memory_viz import (
    load_memory_call_log,
    index_memory_call_log,
    create_memory_database_viz,
)

# Re-export for backwards compatibility
__all__ = [
    "get_edges",
    "generate_layout",
    "normalize_layout",
    "get_completed_stages",
    "get_stage_dir_map",
    "stage_dir_to_stage_id",
    "cfg_to_tree_struct",
    "generate_html",
    "generate",
    "create_unified_viz",
    "create_memory_database_viz",
]


def cfg_to_tree_struct(cfg, jou: Journal, out_path: Path = None):
    """Convert journal to tree structure data for visualization.

    Args:
        cfg: Configuration object
        jou: Journal containing nodes
        out_path: Optional output path for determining log directory

    Returns:
        Dictionary containing tree structure data for visualization
    """
    # Check if there are any root nodes that should connect to the virtual None root
    # (i.e., nodes with parent=None and no inherited_from_node_id)
    has_true_root_nodes = any(
        n.parent is None and not getattr(n, "inherited_from_node_id", None)
        for n in jou
    )

    # Include None root node only if there are true root nodes to connect to it
    # The None node is at index len(jou)
    include_none_root = has_true_root_nodes
    edges = list(get_edges(jou, include_none_root=include_none_root))
    n_nodes = len(jou) + (1 if include_none_root else 0)
    print(f"[red]Edges (with None root={include_none_root}): {edges}[/red]")

    try:
        gen_layout = generate_layout(n_nodes, edges)
    except Exception as e:
        print(f"Error in generate_layout: {e}")
        raise

    try:
        layout = normalize_layout(gen_layout)
    except Exception as e:
        print(f"Error in normalize_layout: {e}")
        raise

    best_node = jou.get_best_node(cfg=cfg)
    metrics = []
    is_best_node = []

    for n in jou:
        if n.metric:
            # Pass the entire metric structure for the new format
            if isinstance(n.metric.value, dict) and "metric_names" in n.metric.value:
                metrics.append(n.metric.value)
            else:
                # Handle legacy format by wrapping it in the new structure
                metrics.append(
                    {
                        "metric_names": [
                            {
                                "metric_name": n.metric.name or "value",
                                "lower_is_better": not n.metric.maximize,
                                "description": n.metric.description or "",
                                "data": [
                                    {
                                        "dataset_name": "default",
                                        "value": n.metric.value,
                                    }
                                ],
                            }
                        ]
                    }
                )
        else:
            metrics.append(None)

        # Track whether this is the best node
        is_best_node.append(n is best_node)

    # Build tree structure data
    tmp = _build_tree_data(cfg, jou, edges, layout, metrics, is_best_node)

    # Add stage and memory information if output path is provided
    if out_path:
        _add_stage_info(tmp, jou, out_path)

    # Append None root node entries to all list fields
    # The None root node is at index len(jou) and represents the virtual root
    if include_none_root:
        _append_none_root_entries(tmp)

    return tmp


def _build_tree_data(cfg, jou: Journal, edges, layout, metrics, is_best_node) -> dict:
    """Build the core tree data structure."""
    tmp = {}

    # Core structure data
    tmp["edges"] = edges
    tmp["layout"] = layout.tolist()
    tmp["exp_name"] = cfg.exp_name
    tmp["metrics"] = metrics
    tmp["is_best_node"] = is_best_node

    # Node data - extracted with error handling
    node_fields = [
        ("plan", lambda n: textwrap.fill(str(n.plan) if n.plan is not None else "", width=80)),
        ("code", lambda n: n.code),
        ("term_out", lambda n: textwrap.fill(str(n._term_out) if n._term_out is not None else "", width=80)),
        ("analysis", lambda n: textwrap.fill(str(n.analysis) if n.analysis is not None else "", width=80)),
        ("node_id", lambda n: n.id),
        ("branch_id", lambda n: getattr(n, "branch_id", None)),
        ("inherited_from_node_id", lambda n: getattr(n, "inherited_from_node_id", None)),
        ("exc_type", lambda n: n.exc_type),
        ("exc_info", lambda n: n.exc_info),
        ("exc_stack", lambda n: n.exc_stack),
        ("plots", lambda n: n.plots),
        ("plot_paths", lambda n: n.plot_paths),
        ("plot_analyses", lambda n: n.plot_analyses),
        ("vlm_feedback_summary", lambda n: textwrap.fill(str(n.vlm_feedback_summary) if n.vlm_feedback_summary is not None else "", width=80)),
        ("exec_time", lambda n: n.exec_time),
        ("exec_time_feedback", lambda n: textwrap.fill(str(n.exec_time_feedback) if n.exec_time_feedback is not None else "", width=80)),
        ("datasets_successfully_tested", lambda n: n.datasets_successfully_tested),
        ("plot_code", lambda n: n.plot_code),
        ("plot_plan", lambda n: n.plot_plan),
        ("ablation_name", lambda n: n.ablation_name),
        ("hyperparam_name", lambda n: n.hyperparam_name),
        ("is_seed_node", lambda n: n.is_seed_node),
        ("is_seed_agg_node", lambda n: n.is_seed_agg_node),
        ("parse_metrics_plan", lambda n: textwrap.fill(str(n.parse_metrics_plan) if n.parse_metrics_plan is not None else "", width=80)),
        ("parse_metrics_code", lambda n: n.parse_metrics_code),
        ("parse_term_out", lambda n: textwrap.fill(str(n.parse_term_out) if n.parse_term_out is not None else "", width=80)),
        ("parse_exc_type", lambda n: n.parse_exc_type),
        ("parse_exc_info", lambda n: n.parse_exc_info),
        ("parse_exc_stack", lambda n: n.parse_exc_stack),
    ]

    for field_name, extractor in node_fields:
        try:
            tmp[field_name] = [extractor(n) for n in jou]
        except Exception as e:
            print(f"Error setting {field_name}: {e}")
            raise

    return tmp


def _add_stage_info(tmp: dict, jou: Journal, out_path: Path):
    """Add stage and memory information to tree data."""
    log_dir = out_path.parent.parent
    tmp["completed_stages"] = get_completed_stages(log_dir)
    tmp["stage_dir_map"] = get_stage_dir_map(log_dir)
    tmp["log_dir_path"] = os.path.relpath(log_dir, out_path.parent)

    stage_id = stage_dir_to_stage_id(out_path.parent.name)
    if stage_id:
        tmp["current_stage"] = stage_id

    try:
        memory_events = load_memory_call_log(log_dir)
        if memory_events:
            tmp["memory_events"] = index_memory_call_log(
                memory_events,
                tmp.get("node_id") or [n.id for n in jou],
                tmp.get("branch_id") or [getattr(n, "branch_id", None) for n in jou],
            )
        else:
            tmp["memory_events"] = [[] for _ in jou]
    except Exception as e:
        print(f"Error setting memory_events: {e}")
        tmp["memory_events"] = [[] for _ in jou]


def _append_none_root_entries(tmp: dict):
    """Append None root node entries to all list fields."""
    none_node_defaults = {
        "plan": "None (Virtual Root)",
        "code": "",
        "term_out": "",
        "analysis": "",
        "node_id": "none_root",
        "branch_id": None,
        "inherited_from_node_id": None,
        "exc_type": None,
        "exc_info": None,
        "exc_stack": None,
        "metrics": None,
        "is_best_node": False,
        "plots": [],
        "plot_paths": [],
        "plot_analyses": [],
        "vlm_feedback_summary": "",
        "exec_time": None,
        "exec_time_feedback": "",
        "datasets_successfully_tested": [],
        "plot_code": None,
        "plot_plan": None,
        "ablation_name": None,
        "hyperparam_name": None,
        "is_seed_node": False,
        "is_seed_agg_node": False,
        "parse_metrics_plan": "",
        "parse_metrics_code": "",
        "parse_term_out": "",
        "parse_exc_type": None,
        "parse_exc_info": None,
        "parse_exc_stack": None,
    }

    for key, default_value in none_node_defaults.items():
        if key in tmp and isinstance(tmp[key], list):
            tmp[key].append(default_value)

    # Also append empty list for memory_events if it exists
    if "memory_events" in tmp and isinstance(tmp["memory_events"], list):
        tmp["memory_events"].append([])


def generate_html(tree_graph_str: str) -> str:
    """Generate HTML content from tree graph data.

    Args:
        tree_graph_str: JSON string of tree graph data

    Returns:
        Complete HTML content as string
    """
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace('"PLACEHOLDER_TREE_DATA"', tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    """Generate tree visualization and related files.

    Args:
        cfg: Configuration object
        jou: Journal containing nodes
        out_path: Output path for the HTML file
    """
    print("[red]Checking Journal[/red]")
    try:
        tree_struct = cfg_to_tree_struct(cfg, jou, out_path)
    except Exception as e:
        print(f"Error in cfg_to_tree_struct: {e}")
        raise

    # Save tree data as JSON for loading by the tabbed visualization
    try:
        data_path = out_path.parent / "tree_data.json"
        with open(data_path, "w") as f:
            json.dump(tree_struct, f)
    except Exception as e:
        print(f"Error saving tree data JSON: {e}")

    try:
        tree_graph_str = json.dumps(tree_struct)
    except Exception as e:
        print(f"Error in json.dumps: {e}")
        raise

    try:
        html = generate_html(tree_graph_str)
    except Exception as e:
        print(f"Error in generate_html: {e}")
        raise

    with open(out_path, "w") as f:
        f.write(html)

    # Create a unified tree visualization that shows all stages
    try:
        create_unified_viz(cfg, out_path)
    except Exception as e:
        print(f"Error creating unified visualization: {e}")
        # Continue even if unified viz creation fails

    # Create memory database visualization
    try:
        create_memory_database_viz(cfg, out_path)
    except Exception as e:
        print(f"Error creating memory database visualization: {e}")
        # Continue even if memory database viz creation fails


def create_unified_viz(cfg, current_stage_viz_path: Path):
    """
    Create a unified visualization that shows all completed stages in a tabbed interface.
    This will be placed in the main log directory.

    Args:
        cfg: Configuration object
        current_stage_viz_path: Path to the current stage visualization file
    """
    # The main log directory is two levels up from the stage-specific visualization
    log_dir = current_stage_viz_path.parent.parent

    # Get the current stage name from the path
    current_stage = stage_dir_to_stage_id(current_stage_viz_path.parent.name)

    # Create a combined visualization at the top level
    unified_viz_path = log_dir / "unified_tree_viz.html"

    # Copy the template files
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.html") as f:
        html = f.read()

    with open(template_dir / "template.js") as f:
        js = f.read()

    # Get completed stages by checking directories
    completed_stages = get_completed_stages(log_dir)
    stage_dir_map = get_stage_dir_map(log_dir)
    if not current_stage:
        current_stage = completed_stages[0] if completed_stages else "Stage_1"

    base_data = {
        "current_stage": current_stage,
        "completed_stages": completed_stages,
        "stage_dir_map": stage_dir_map,
        "log_dir_path": ".",
    }

    # Replace the placeholder in the JS with our data
    js = js.replace('"PLACEHOLDER_TREE_DATA"', json.dumps(base_data))

    # Replace the placeholder in the HTML with our JS
    html = html.replace("<!-- placeholder -->", js)

    # Write the unified visualization
    with open(unified_viz_path, "w") as f:
        f.write(html)

    print(f"[green]Created unified visualization at {unified_viz_path}[/green]")
