"""Export journal to HTML visualization of tree + code."""

import json
import os
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal

from rich import print


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def get_completed_stages(log_dir):
    """
    Determine completed stages by checking for the existence of stage directories
    that contain evidence of completion (tree_data.json, tree_plot.html, or journal.json).

    Returns:
        list: A list of stage names (e.g., ["Stage_1", "Stage_2"])
    """
    completed_stages = []

    # Check for each stage (1-4)
    for stage_num in range(1, 5):
        prefix = f"stage_{stage_num}"

        # Find all directories that match this stage number
        matching_dirs = [
            d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        ]

        # Check if any of these directories have completion evidence
        for stage_dir in matching_dirs:
            has_tree_data = (stage_dir / "tree_data.json").exists()
            has_tree_plot = (stage_dir / "tree_plot.html").exists()
            has_journal = (stage_dir / "journal.json").exists()

            if has_tree_data or has_tree_plot or has_journal:
                # Found evidence this stage was completed
                completed_stages.append(f"Stage_{stage_num}")
                break  # No need to check other directories for this stage

    return completed_stages


def get_stage_dir_map(log_dir: Path) -> dict[str, str]:
    """
    Map Stage_X to the actual stage directory name under log_dir.
    Picks the newest directory when multiple matches exist for a stage.
    """
    stage_map: dict[str, str] = {}
    for stage_dir in log_dir.iterdir():
        if not stage_dir.is_dir():
            continue
        if not stage_dir.name.startswith("stage_"):
            continue
        parts = stage_dir.name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        stage_id = f"Stage_{parts[1]}"
        try:
            mtime = stage_dir.stat().st_mtime
        except OSError:
            mtime = 0.0
        previous = stage_map.get(stage_id)
        if not previous:
            stage_map[stage_id] = stage_dir.name
            continue
        try:
            previous_mtime = (log_dir / previous).stat().st_mtime
        except OSError:
            previous_mtime = 0.0
        if mtime >= previous_mtime:
            stage_map[stage_id] = stage_dir.name
    return stage_map


def stage_dir_to_stage_id(stage_dir_name: str) -> str | None:
    if not stage_dir_name.startswith("stage_"):
        return None
    parts = stage_dir_name.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return f"Stage_{parts[1]}"


def _load_memory_events(log_dir: Path) -> list[dict]:
    candidates = [
        log_dir / "memory" / "memory_calls.jsonl",
        log_dir / "memory_calls.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        events: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        events.append(payload)
        except Exception:
            continue
        if events:
            return events
    return []


def _index_memory_events(
    events: list[dict],
    node_ids: list[str],
    branch_ids: list[str | None],
) -> list[list[dict]]:
    indexed: list[list[dict]] = [[] for _ in node_ids]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids) if node_id}
    branch_index = {
        branch_id: idx for idx, branch_id in enumerate(branch_ids) if branch_id
    }
    for event in events:
        if not isinstance(event, dict):
            continue
        node_id = event.get("node_id")
        branch_id = event.get("branch_id")
        idx = None
        if node_id and node_id in node_index:
            idx = node_index[node_id]
        elif branch_id and branch_id in branch_index:
            idx = branch_index[branch_id]
        if idx is None:
            continue
        indexed[idx].append(event)
    for entries in indexed:
        entries.sort(key=lambda entry: entry.get("ts", 0) or 0)
    return indexed


def cfg_to_tree_struct(cfg, jou: Journal, out_path: Path = None):
    edges = list(get_edges(jou))
    print(f"[red]Edges: {edges}[/red]")
    try:
        gen_layout = generate_layout(len(jou), edges)
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
        # print(f"Node {n.id} exc_stack: {type(n.exc_stack)} = {n.exc_stack}")
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
                                        "final_value": n.metric.value,
                                        "best_value": n.metric.value,
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

    tmp = {}

    # Add each item individually with error handling
    try:
        tmp["edges"] = edges
    except Exception as e:
        print(f"Error setting edges: {e}")
        raise

    try:
        tmp["layout"] = layout.tolist()
    except Exception as e:
        print(f"Error setting layout: {e}")
        raise

    try:
        tmp["plan"] = [
            textwrap.fill(str(n.plan) if n.plan is not None else "", width=80)
            for n in jou.nodes
        ]
    except Exception as e:
        print(f"Error setting plan: {e}")
        raise

    try:
        tmp["code"] = [n.code for n in jou]
    except Exception as e:
        print(f"Error setting code: {e}")
        raise

    try:
        tmp["term_out"] = [
            textwrap.fill(str(n._term_out) if n._term_out is not None else "", width=80)
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting term_out: {e}")
        print(f"n.term_out: {n._term_out}")
        raise

    try:
        tmp["analysis"] = [
            textwrap.fill(str(n.analysis) if n.analysis is not None else "", width=80)
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting analysis: {e}")
        raise

    try:
        tmp["node_id"] = [n.id for n in jou]
    except Exception as e:
        print(f"Error setting node_id: {e}")
        raise

    try:
        tmp["branch_id"] = [getattr(n, "branch_id", None) for n in jou]
    except Exception as e:
        print(f"Error setting branch_id: {e}")
        raise

    try:
        tmp["exc_type"] = [n.exc_type for n in jou]
    except Exception as e:
        print(f"Error setting exc_type: {e}")
        raise

    try:
        tmp["exc_info"] = [n.exc_info for n in jou]
    except Exception as e:
        print(f"Error setting exc_info: {e}")
        raise

    try:
        tmp["exc_stack"] = [n.exc_stack for n in jou]
    except Exception as e:
        print(f"Error setting exc_stack: {e}")
        raise

    try:
        tmp["exp_name"] = cfg.exp_name
    except Exception as e:
        print(f"Error setting exp_name: {e}")
        raise

    try:
        tmp["metrics"] = metrics
    except Exception as e:
        print(f"Error setting metrics: {e}")
        raise

    try:
        tmp["is_best_node"] = is_best_node
    except Exception as e:
        print(f"Error setting is_best_node: {e}")
        raise

    try:
        tmp["plots"] = [n.plots for n in jou]
    except Exception as e:
        print(f"Error setting plots: {e}")
        raise

    try:
        tmp["plot_paths"] = [n.plot_paths for n in jou]
    except Exception as e:
        print(f"Error setting plot_paths: {e}")
        raise

    try:
        tmp["plot_analyses"] = [n.plot_analyses for n in jou]
    except Exception as e:
        print(f"Error setting plot_analyses: {e}")
        raise

    try:
        tmp["vlm_feedback_summary"] = [
            textwrap.fill(
                (
                    str(n.vlm_feedback_summary)
                    if n.vlm_feedback_summary is not None
                    else ""
                ),
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting vlm_feedback_summary: {e}")
        raise

    try:
        tmp["exec_time"] = [n.exec_time for n in jou]
    except Exception as e:
        print(f"Error setting exec_time: {e}")
        raise

    try:
        tmp["exec_time_feedback"] = [
            textwrap.fill(
                str(n.exec_time_feedback) if n.exec_time_feedback is not None else "",
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting exec_time_feedback: {e}")
        raise

    try:
        tmp["datasets_successfully_tested"] = [
            n.datasets_successfully_tested for n in jou
        ]
    except Exception as e:
        print(f"Error setting datasets_successfully_tested: {e}")
        raise

    try:
        tmp["plot_code"] = [n.plot_code for n in jou]
    except Exception as e:
        print(f"Error setting plot_code: {e}")
        raise

    try:
        tmp["plot_plan"] = [n.plot_plan for n in jou]
    except Exception as e:
        print(f"Error setting plot_plan: {e}")
        raise

    try:
        tmp["ablation_name"] = [n.ablation_name for n in jou]
    except Exception as e:
        print(f"Error setting ablation_name: {e}")
        raise

    try:
        tmp["hyperparam_name"] = [n.hyperparam_name for n in jou]
    except Exception as e:
        print(f"Error setting hyperparam_name: {e}")
        raise

    try:
        tmp["is_seed_node"] = [n.is_seed_node for n in jou]
    except Exception as e:
        print(f"Error setting is_seed_node: {e}")
        raise

    try:
        tmp["is_seed_agg_node"] = [n.is_seed_agg_node for n in jou]
    except Exception as e:
        print(f"Error setting is_seed_agg_node: {e}")
        raise

    try:
        tmp["parse_metrics_plan"] = [
            textwrap.fill(
                str(n.parse_metrics_plan) if n.parse_metrics_plan is not None else "",
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting parse_metrics_plan: {e}")
        raise

    try:
        tmp["parse_metrics_code"] = [n.parse_metrics_code for n in jou]
    except Exception as e:
        print(f"Error setting parse_metrics_code: {e}")
        raise

    try:
        tmp["parse_term_out"] = [
            textwrap.fill(
                str(n.parse_term_out) if n.parse_term_out is not None else "", width=80
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting parse_term_out: {e}")
        raise

    try:
        tmp["parse_exc_type"] = [n.parse_exc_type for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_type: {e}")
        raise

    try:
        tmp["parse_exc_info"] = [n.parse_exc_info for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_info: {e}")
        raise

    try:
        tmp["parse_exc_stack"] = [n.parse_exc_stack for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_stack: {e}")
        raise

    # Add the list of completed stages by checking directories
    if out_path:
        log_dir = out_path.parent.parent
        tmp["completed_stages"] = get_completed_stages(log_dir)
        tmp["stage_dir_map"] = get_stage_dir_map(log_dir)
        tmp["log_dir_path"] = os.path.relpath(log_dir, out_path.parent)
        stage_id = stage_dir_to_stage_id(out_path.parent.name)
        if stage_id:
            tmp["current_stage"] = stage_id
        try:
            memory_events = _load_memory_events(log_dir)
            if memory_events:
                tmp["memory_events"] = _index_memory_events(
                    memory_events,
                    tmp.get("node_id") or [n.id for n in jou],
                    tmp.get("branch_id") or [getattr(n, "branch_id", None) for n in jou],
                )
            else:
                tmp["memory_events"] = [[] for _ in jou]
        except Exception as e:
            print(f"Error setting memory_events: {e}")
            tmp["memory_events"] = [[] for _ in jou]

    return tmp


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace('"PLACEHOLDER_TREE_DATA"', tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    print("[red]Checking Journal[/red]")
    try:
        tree_struct = cfg_to_tree_struct(cfg, jou, out_path)
    except Exception as e:
        print(f"Error in cfg_to_tree_struct: {e}")
        raise

    # Save tree data as JSON for loading by the tabbed visualization
    try:
        # Save the tree data as a JSON file in the same directory
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


def create_unified_viz(cfg, current_stage_viz_path):
    """
    Create a unified visualization that shows all completed stages in a tabbed interface.
    This will be placed in the main log directory.
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
