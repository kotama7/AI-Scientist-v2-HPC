"""Export journal to HTML visualization of tree + code."""

import json
import os
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from igraph import Graph

from rich import print

from ..journal import Journal


def get_edges(journal: Journal, include_none_root: bool = False):
    """Generate edges from parent-child relationships.

    If include_none_root is True, adds a virtual "None" root node at index len(journal)
    and connects all root nodes (nodes with parent=None and no inherited_from_node_id) to it.
    Nodes with inherited_from_node_id are excluded from the None root connection
    as they represent cross-stage inheritance from a previous stage's best node.
    """
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)

    if include_none_root:
        # Add edges from None root (at index len(journal)) to all root nodes
        # Exclude nodes that are inherited from a previous stage
        none_root_idx = len(journal)
        for node in journal:
            if node.parent is None and not getattr(node, "inherited_from_node_id", None):
                yield (none_root_idx, node.step)


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
        tmp["inherited_from_node_id"] = [getattr(n, "inherited_from_node_id", None) for n in jou]
    except Exception as e:
        print(f"Error setting inherited_from_node_id: {e}")
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

    # Append None root node entries to all list fields
    # The None root node is at index len(jou) and represents the virtual root
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

    # Only append None root node entries if we're including the virtual root
    if include_none_root:
        for key, default_value in none_node_defaults.items():
            if key in tmp and isinstance(tmp[key], list):
                tmp[key].append(default_value)

        # Also append empty list for memory_events if it exists
        if "memory_events" in tmp and isinstance(tmp["memory_events"], list):
            tmp["memory_events"].append([])

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

    # Create memory database visualization
    try:
        create_memory_database_viz(cfg, out_path)
    except Exception as e:
        print(f"Error creating memory database visualization: {e}")
        # Continue even if memory database viz creation fails


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


def _load_journal_nodes_from_tree_data(log_dir: Path) -> list[dict]:
    """Load node information from all tree_data.json files in stage directories."""
    journal_nodes = []
    seen_node_ids = set()

    # Find all tree_data.json files in stage directories
    for stage_dir in log_dir.iterdir():
        if not stage_dir.is_dir() or not stage_dir.name.startswith("stage_"):
            continue
        tree_data_path = stage_dir / "tree_data.json"
        if not tree_data_path.exists():
            continue

        try:
            with open(tree_data_path, "r", encoding="utf-8") as f:
                tree_data = json.load(f)

            node_ids = tree_data.get("node_id", [])
            edges = tree_data.get("edges", [])

            # Build parent mapping from edges
            # edges are in format [[parent_idx, child_idx], ...]
            parent_map = {}
            for edge in edges:
                if len(edge) == 2:
                    parent_idx, child_idx = edge
                    if child_idx < len(node_ids) and parent_idx < len(node_ids):
                        parent_map[child_idx] = parent_idx

            for idx, node_id in enumerate(node_ids):
                if node_id in seen_node_ids or node_id == "none_root":
                    continue
                seen_node_ids.add(node_id)

                parent_idx = parent_map.get(idx)
                parent_id = None
                if parent_idx is not None and parent_idx < len(node_ids):
                    parent_node_id = node_ids[parent_idx]
                    if parent_node_id != "none_root":
                        parent_id = parent_node_id

                journal_nodes.append({
                    "id": node_id,
                    "parent_id": parent_id,
                    "node_uid": node_id,
                    "created_at": None,  # Unknown from tree_data.json
                    "source": "journal",
                })
        except Exception as e:
            print(f"[yellow]Warning: Could not load tree_data.json from {stage_dir}: {e}[/yellow]")

    return journal_nodes


def create_memory_database_viz(cfg, current_stage_viz_path: Path):
    """
    Create a memory database visualization HTML file.
    This will be placed in the main log directory alongside unified_tree_viz.html.
    """
    # The main log directory is two levels up from the stage-specific visualization
    log_dir = current_stage_viz_path.parent.parent
    # workspace_dir contains the actual memory.sqlite (separate from log_dir)
    workspace_dir = Path(cfg.workspace_dir)

    # Find memory.sqlite - check workspace_dir first, then log_dir for backwards compatibility
    possible_paths = [
        workspace_dir / "memory" / "memory.sqlite",
        workspace_dir / "memory.sqlite",
        log_dir / "memory" / "memory.sqlite",
        log_dir / "memory.sqlite",
    ]

    db_path = None
    for p in possible_paths:
        if p.exists():
            db_path = p
            break

    if not db_path:
        print(f"[yellow]Memory database not found in {log_dir}, skipping memory_database.html generation[/yellow]")
        return

    print(f"[blue]Found memory database: {db_path}[/blue]")

    # Load memory data and generate HTML
    try:
        memory_data = _load_memory_database(db_path)

        # Load journal nodes from tree_data.json files and merge with SQLite data
        journal_nodes = _load_journal_nodes_from_tree_data(log_dir)
        if journal_nodes:
            memory_data = _merge_journal_nodes_into_memory_data(memory_data, journal_nodes)
            print(f"[blue]Merged {len(journal_nodes)} journal nodes into memory data[/blue]")

        output_path = log_dir / "memory_database.html"
        experiment_name = log_dir.name
        _generate_memory_database_html(memory_data, output_path, experiment_name)
        print(f"[green]Created memory database visualization at {output_path}[/green]")
    except Exception as e:
        print(f"[red]Error generating memory database visualization: {e}[/red]")


def _merge_journal_nodes_into_memory_data(memory_data: dict, journal_nodes: list[dict]) -> dict:
    """Merge journal nodes into memory data, adding missing branches and fixing parent relationships.

    This function:
    1. Adds missing branches from journal_nodes that don't exist in memory_data
    2. Updates parent_id for existing branches to match tree_data.json (the source of truth for tree structure)
    """
    existing_branch_ids = {b["id"] for b in memory_data["branches"]}
    existing_node_uids = {b["node_uid"] for b in memory_data["branches"]}

    # Build node_uid -> branch mapping for quick lookup
    node_uid_to_branch = {b["node_uid"]: b for b in memory_data["branches"] if b["node_uid"]}
    branch_id_to_branch = {b["id"]: b for b in memory_data["branches"]}

    # Find root branch for orphan nodes
    root_branch = next((b for b in memory_data["branches"] if b["node_uid"] == "root"), None)
    root_branch_id = root_branch["id"] if root_branch else None

    # Build correct parent map from journal_nodes (tree_data.json is the source of truth)
    # journal_node["id"] == journal_node["node_uid"] == the node's unique identifier
    correct_parent_map = {}  # child_node_uid -> parent_node_uid
    for jnode in journal_nodes:
        node_id = jnode["id"]
        parent_id = jnode.get("parent_id")
        if parent_id:
            correct_parent_map[node_id] = parent_id

    # First pass: Add all missing branches (without resolving parent yet)
    new_branches = []
    for jnode in journal_nodes:
        node_id = jnode["id"]
        # Skip if already exists (by id or node_uid)
        if node_id in existing_branch_ids or node_id in existing_node_uids:
            continue

        new_branches.append({
            "id": node_id,
            "parent_id": None,  # Will be resolved in second pass
            "node_uid": node_id,
            "created_at": jnode.get("created_at"),
        })
        existing_branch_ids.add(node_id)
        existing_node_uids.add(node_id)
        node_uid_to_branch[node_id] = new_branches[-1]
        branch_id_to_branch[node_id] = new_branches[-1]

    # Add new branches to memory data
    if new_branches:
        memory_data["branches"].extend(new_branches)

    # Second pass: Fix parent_id for all branches based on tree_data.json
    # This ensures the tree structure matches what's in tree_data.json
    for branch in memory_data["branches"]:
        node_uid = branch.get("node_uid")
        if not node_uid or node_uid == "root":
            continue

        correct_parent_uid = correct_parent_map.get(node_uid)
        if correct_parent_uid:
            # Find the parent branch (by node_uid first, then by id)
            parent_branch = node_uid_to_branch.get(correct_parent_uid) or branch_id_to_branch.get(correct_parent_uid)
            if parent_branch:
                new_parent_id = parent_branch["id"]
                if branch["parent_id"] != new_parent_id:
                    branch["parent_id"] = new_parent_id
            else:
                # Parent not found - keep existing or use root
                if not branch["parent_id"]:
                    branch["parent_id"] = root_branch_id
        elif not branch["parent_id"]:
            # No parent specified in tree_data and no existing parent - attach to root
            branch["parent_id"] = root_branch_id

    # Sort by created_at (None values go last)
    memory_data["branches"].sort(key=lambda b: (b["created_at"] is None, b["created_at"] or 0))

    return memory_data


def _load_memory_database(db_path: Path) -> dict:
    """Load all memory data from SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    def parse_tags(tags_str: str) -> list[str]:
        if not tags_str:
            return []
        try:
            return json.loads(tags_str)
        except json.JSONDecodeError:
            return []

    def extract_phase_from_tags(tags: list[str]) -> str | None:
        for tag in tags:
            if tag.startswith("phase:"):
                return tag[6:]
        return None

    # Load branches (tree structure)
    cursor.execute("SELECT id, parent_id, node_uid, created_at FROM branches ORDER BY created_at")
    branches = []
    for row in cursor.fetchall():
        branches.append({
            "id": row["id"],
            "parent_id": row["parent_id"],
            "node_uid": row["node_uid"],
            "created_at": row["created_at"],
        })

    # Load core_kv (Working Context)
    cursor.execute("SELECT branch_id, key, value, updated_at FROM core_kv ORDER BY updated_at")
    core_kv = defaultdict(list)
    for row in cursor.fetchall():
        core_kv[row["branch_id"]].append({
            "key": row["key"],
            "value": row["value"],
            "updated_at": row["updated_at"],
        })

    # Load events (Recall Storage)
    cursor.execute("SELECT id, branch_id, kind, text, tags, created_at, task_hint, memory_size FROM events ORDER BY created_at")
    events = defaultdict(list)
    for row in cursor.fetchall():
        tags = parse_tags(row["tags"])
        phase = extract_phase_from_tags(tags)
        event_data = {
            "id": row["id"],
            "branch_id": row["branch_id"],
            "kind": row["kind"],
            "text": row["text"],
            "tags": tags,
            "phase": phase,
            "created_at": row["created_at"],
            "task_hint": row["task_hint"],
            "memory_size": row["memory_size"],
        }
        events[row["branch_id"]].append(event_data)

    # Load archival (Archive Storage)
    cursor.execute("SELECT id, branch_id, text, tags, created_at FROM archival ORDER BY created_at")
    archival = defaultdict(list)
    for row in cursor.fetchall():
        tags = parse_tags(row["tags"])
        phase = None
        for tag in tags:
            if "PHASE0" in tag:
                phase = "phase0"
            elif "PHASE1" in tag:
                phase = "phase1"
            elif "PHASE2" in tag:
                phase = "phase2"
            elif "PHASE3" in tag:
                phase = "phase3"
            elif "PHASE4" in tag:
                phase = "phase4"

        archival_data = {
            "id": row["id"],
            "branch_id": row["branch_id"],
            "text": row["text"],
            "tags": tags,
            "phase": phase,
            "created_at": row["created_at"],
        }
        archival[row["branch_id"]].append(archival_data)

    conn.close()

    return {
        "branches": branches,
        "core_kv": dict(core_kv),
        "events": dict(events),
        "archival": dict(archival),
    }


def _build_memory_tree_layout(branches: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Build tree layout from branches."""
    if not branches:
        return [], []

    children = defaultdict(list)
    node_to_idx = {}
    id_to_node_uid = {}  # Map id -> node_uid for lookup

    for i, branch in enumerate(branches):
        node_to_idx[branch["id"]] = i
        # Also index by node_uid if different from id (for parent_id resolution)
        if branch.get("node_uid") and branch["node_uid"] != branch["id"]:
            node_to_idx[branch["node_uid"]] = i
            id_to_node_uid[branch["id"]] = branch["node_uid"]
        if branch["parent_id"]:
            children[branch["parent_id"]].append(branch["id"])

    root_nodes = [b["id"] for b in branches if not b["parent_id"]]
    layout = [(0.0, 0.0)] * len(branches)
    edges = []

    def get_children(node_id: str) -> list:
        """Get children by checking both id and node_uid."""
        child_list = children.get(node_id, [])
        # Also check node_uid if this id has a different node_uid
        node_uid = id_to_node_uid.get(node_id)
        if node_uid:
            child_list = child_list + children.get(node_uid, [])
        return child_list

    def calc_subtree_width(node_id: str) -> int:
        child_list = get_children(node_id)
        if not child_list:
            return 1
        return sum(calc_subtree_width(c) for c in child_list)

    def layout_subtree(node_id: str, x: float, y: float, width: float):
        idx = node_to_idx.get(node_id)
        if idx is None:
            return

        layout[idx] = (x + width / 2, y)

        child_list = get_children(node_id)
        if not child_list:
            return

        total_child_width = sum(calc_subtree_width(c) for c in child_list)
        current_x = x

        for child_id in child_list:
            child_width = (calc_subtree_width(child_id) / total_child_width) * width
            layout_subtree(child_id, current_x, y + 0.15, child_width)

            parent_idx = node_to_idx.get(node_id)
            child_idx = node_to_idx.get(child_id)
            if parent_idx is not None and child_idx is not None:
                edges.append((parent_idx, child_idx))

            current_x += child_width

    if root_nodes:
        total_width = sum(calc_subtree_width(r) for r in root_nodes)
        current_x = 0.0
        for root_id in root_nodes:
            root_width = (calc_subtree_width(root_id) / total_width)
            layout_subtree(root_id, current_x, 0.05, root_width)
            current_x += root_width

    if layout:
        min_x = min(p[0] for p in layout)
        max_x = max(p[0] for p in layout)
        min_y = min(p[1] for p in layout)
        max_y = max(p[1] for p in layout)

        x_range = max_x - min_x if max_x > min_x else 1
        y_range = max_y - min_y if max_y > min_y else 1

        layout = [
            ((p[0] - min_x) / x_range * 0.9 + 0.05, (p[1] - min_y) / y_range * 0.85 + 0.05)
            for p in layout
        ]

    return layout, edges


def _get_phases_for_branch(branch_id: str, events: dict, archival: dict) -> list[str]:
    """Get all phases associated with a branch."""
    phases = set()

    for event in events.get(branch_id, []):
        if event.get("phase"):
            phases.add(event["phase"])

    for record in archival.get(branch_id, []):
        if record.get("phase"):
            phases.add(record["phase"])

    if events.get(branch_id) or archival.get(branch_id):
        phases.add("summary")

    phase_order = ["phase0", "phase1", "phase2", "phase3", "phase4", "summary"]
    sorted_phases = []
    for p in phase_order:
        if p in phases:
            sorted_phases.append(p)
    for p in sorted(phases):
        if p not in sorted_phases:
            sorted_phases.append(p)

    return sorted_phases


def _get_phases_for_branch_accumulated(events: list[dict], archival: list[dict]) -> list[str]:
    """Get all phases from accumulated events and archival data."""
    phases = set()

    for event in events:
        if event.get("phase"):
            phases.add(event["phase"])

    for record in archival:
        if record.get("phase"):
            phases.add(record["phase"])

    if events or archival:
        phases.add("summary")

    phase_order = ["phase0", "phase1", "phase2", "phase3", "phase4", "summary"]
    sorted_phases = []
    for p in phase_order:
        if p in phases:
            sorted_phases.append(p)
    for p in sorted(phases):
        if p not in sorted_phases:
            sorted_phases.append(p)

    return sorted_phases


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _get_ancestor_chain(branch_id: str, branches: list[dict]) -> list[dict]:
    """Get the chain of ancestors for a branch (from root to parent, excluding self)."""
    branch_map = {b["id"]: b for b in branches}
    # Also map by node_uid for lookups
    for b in branches:
        if b.get("node_uid") and b["node_uid"] != b["id"]:
            branch_map[b["node_uid"]] = b

    ancestors = []
    current = branch_map.get(branch_id)
    if not current:
        return []

    parent_id = current.get("parent_id")
    while parent_id:
        parent = branch_map.get(parent_id)
        if not parent:
            break
        ancestors.append(parent)
        parent_id = parent.get("parent_id")

    # Reverse to get root-to-parent order
    ancestors.reverse()
    return ancestors


def _collect_inherited_data(
    branch_id: str,
    branches: list[dict],
    core_kv: dict,
    events: dict,
    archival: dict,
    branch_to_index: dict,
    exclude_virtual_root: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Collect inherited data from ancestors.

    Args:
        branch_id: The branch to collect ancestors for.
        branches: List of all branches.
        core_kv: Core key-value data by branch_id.
        events: Events data by branch_id.
        archival: Archival data by branch_id.
        branch_to_index: Mapping of branch_id to index.
        exclude_virtual_root: If True, exclude data from virtual root nodes (node_uid == "root").

    Returns:
        Tuple of (inherited_core_kv, inherited_events, inherited_archival, ancestors_info)
    """
    ancestors = _get_ancestor_chain(branch_id, branches)

    inherited_core_kv = []
    inherited_events = []
    inherited_archival = []
    ancestors_info = []

    # Track keys already seen (for core_kv, later values override earlier ones)
    seen_keys = set()

    # Collect data from each ancestor (root to parent order)
    for ancestor in ancestors:
        ancestor_id = ancestor["id"]
        ancestor_node_uid = ancestor.get("node_uid")
        ancestor_index = branch_to_index.get(ancestor_id, -1)

        # Skip virtual root's data if requested
        is_virtual = ancestor_node_uid == "root" or ancestor_node_uid == "none_root"

        ancestors_info.append({
            "index": ancestor_index,
            "branch_id": ancestor_id,
            "node_uid": ancestor_node_uid or ancestor_id,
            "is_virtual": is_virtual,
        })

        # Skip collecting data from virtual root
        if exclude_virtual_root and is_virtual:
            continue

        # Collect core_kv (key-value pairs)
        for kv in core_kv.get(ancestor_id, []):
            # For core_kv, we want the most recent value, so track seen keys
            key = kv.get("key")
            if key not in seen_keys:
                inherited_core_kv.append(kv)
                seen_keys.add(key)

        # Collect events
        for event in events.get(ancestor_id, []):
            inherited_events.append(event)

        # Collect archival
        for arch in archival.get(ancestor_id, []):
            inherited_archival.append(arch)

    return inherited_core_kv, inherited_events, inherited_archival, ancestors_info


def _generate_memory_database_html(memory_data: dict, output_path: Path, experiment_name: str = "Memory Database"):
    """Generate the HTML visualization for memory database."""
    branches = memory_data["branches"]
    layout, edges = _build_memory_tree_layout(branches)

    # Build branch_id to index mapping
    branch_to_index = {b["id"]: i for i, b in enumerate(branches)}

    nodes_data = []
    for i, branch in enumerate(branches):
        branch_id = branch["id"]
        node_uid = branch.get("node_uid")
        is_virtual_node = node_uid == "root" or node_uid == "none_root"

        if is_virtual_node:
            # Virtual node (root) should have no memory data
            nodes_data.append({
                "index": i,
                "branch_id": branch_id,
                "node_uid": node_uid,
                "parent_id": branch["parent_id"],
                "phases": [],
                "is_virtual": True,
                # Empty data for virtual node
                "own_core_kv": [],
                "own_events": [],
                "own_archival": [],
                "inherited_core_kv": [],
                "inherited_events": [],
                "inherited_archival": [],
                "ancestors": [],
                # Legacy fields
                "core_kv": [],
                "events": [],
                "archival": [],
            })
            continue

        # Get own data (this branch only)
        own_core_kv = memory_data["core_kv"].get(branch_id, [])
        own_events = memory_data["events"].get(branch_id, [])
        own_archival = memory_data["archival"].get(branch_id, [])

        # Get inherited data from ancestors (excluding virtual root's data)
        inherited_core_kv, inherited_events, inherited_archival, ancestors_info = _collect_inherited_data(
            branch_id,
            branches,
            memory_data["core_kv"],
            memory_data["events"],
            memory_data["archival"],
            branch_to_index,
            exclude_virtual_root=True,
        )

        # Combine own and inherited for phase detection
        all_events = own_events + inherited_events
        all_archival = own_archival + inherited_archival
        phases = _get_phases_for_branch_accumulated(all_events, all_archival)

        nodes_data.append({
            "index": i,
            "branch_id": branch_id,
            "node_uid": node_uid,
            "parent_id": branch["parent_id"],
            "phases": phases,
            "is_virtual": False,
            # Own data (this node only)
            "own_core_kv": own_core_kv,
            "own_events": own_events,
            "own_archival": own_archival,
            # Inherited data (from ancestors)
            "inherited_core_kv": inherited_core_kv,
            "inherited_events": inherited_events,
            "inherited_archival": inherited_archival,
            # Ancestor information
            "ancestors": ancestors_info,
            # Legacy fields for backwards compatibility
            "core_kv": own_core_kv,
            "events": own_events,
            "archival": own_archival,
        })

    js_data = {
        "layout": layout,
        "edges": edges,
        "nodes": nodes_data,
    }

    # Load HTML template from file
    template_path = Path(__file__).parent / "templates" / "memory_database.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html_template = f.read()
    
    # Replace placeholders with actual values
    html_content = html_template.replace("__EXPERIMENT_NAME__", _escape_html(experiment_name))
    html_content = html_content.replace("__JS_DATA__", json.dumps(js_data, ensure_ascii=False))

    output_path.write_text(html_content, encoding='utf-8')
