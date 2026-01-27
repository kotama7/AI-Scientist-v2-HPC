"""Memory database visualization utilities."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich import print


def load_memory_call_log(log_dir: Path) -> list[dict]:
    """Load memory call log from memory_calls.jsonl.

    This file contains all memory operations including:
    - render_for_prompt: Memory reads/injections into prompts
    - mem_recall_append: Recall memory writes
    - mem_node_fork: Node fork operations
    - set_core: Core memory updates
    - write_archival: Archival memory writes

    Args:
        log_dir: The log directory to search for memory_calls.jsonl

    Returns:
        List of memory call event dictionaries
    """
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


def index_memory_call_log(
    events: list[dict],
    node_ids: list[str],
    branch_ids: list[str | None],
) -> list[list[dict]]:
    """Index memory call events by node/branch for tree visualization.

    Args:
        events: List of memory call events
        node_ids: List of node IDs in order
        branch_ids: List of branch IDs in order

    Returns:
        List of event lists, one per node
    """
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


def load_memory_database(db_path: Path) -> dict:
    """Load all memory data from SQLite database.

    Args:
        db_path: Path to the memory.sqlite database

    Returns:
        Dictionary with branches, core_kv, events, and archival data
    """
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
            if tag.startswith("stage:"):
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
    # Check which columns exist (task_hint and memory_size are optional)
    cursor.execute("PRAGMA table_info(events)")
    event_columns = {row[1] for row in cursor.fetchall()}
    has_task_hint = "task_hint" in event_columns
    has_memory_size = "memory_size" in event_columns

    # Build query with available columns
    select_cols = ["id", "branch_id", "kind", "text", "tags", "created_at"]
    if has_task_hint:
        select_cols.append("task_hint")
    if has_memory_size:
        select_cols.append("memory_size")

    cursor.execute(f"SELECT {', '.join(select_cols)} FROM events ORDER BY created_at")
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
            "task_hint": row["task_hint"] if has_task_hint else None,
            "memory_size": row["memory_size"] if has_memory_size else None,
        }
        events[row["branch_id"]].append(event_data)

    # Load archival (Archive Storage)
    cursor.execute("SELECT id, branch_id, text, tags, created_at FROM archival ORDER BY created_at")
    archival = defaultdict(list)
    for row in cursor.fetchall():
        tags = parse_tags(row["tags"])
        phase = extract_phase_from_tags(tags)

        archival_data = {
            "id": row["id"],
            "branch_id": row["branch_id"],
            "text": row["text"],
            "tags": tags,
            "phase": phase,
            "created_at": row["created_at"],
        }
        archival[row["branch_id"]].append(archival_data)

    # Load inherited_exclusions (Copy-on-Write exclusions for inherited events)
    inherited_exclusions: dict[str, set[int]] = defaultdict(set)
    try:
        cursor.execute("SELECT branch_id, excluded_event_id FROM inherited_exclusions")
        for row in cursor.fetchall():
            inherited_exclusions[row["branch_id"]].add(row["excluded_event_id"])
    except sqlite3.OperationalError:
        pass  # Table doesn't exist in older databases

    # Load inherited_summaries (consolidated summaries of inherited events)
    inherited_summaries: dict[str, list[dict]] = defaultdict(list)
    try:
        cursor.execute(
            "SELECT id, branch_id, summary_text, summarized_event_ids, kind, created_at "
            "FROM inherited_summaries ORDER BY created_at"
        )
        for row in cursor.fetchall():
            try:
                summarized_ids = json.loads(row["summarized_event_ids"] or "[]")
            except json.JSONDecodeError:
                summarized_ids = []
            inherited_summaries[row["branch_id"]].append({
                "id": row["id"],
                "summary_text": row["summary_text"],
                "summarized_event_ids": summarized_ids,
                "kind": row["kind"],
                "created_at": row["created_at"],
            })
    except sqlite3.OperationalError:
        pass  # Table doesn't exist in older databases

    conn.close()

    return {
        "branches": branches,
        "core_kv": dict(core_kv),
        "events": dict(events),
        "archival": dict(archival),
        "inherited_exclusions": {k: list(v) for k, v in inherited_exclusions.items()},
        "inherited_summaries": dict(inherited_summaries),
    }


def build_memory_tree_layout(branches: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Build tree layout from branches.

    Args:
        branches: List of branch dictionaries with id, parent_id, node_uid

    Returns:
        Tuple of (layout coordinates, edges)
    """
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


def get_phases_for_branch(branch_id: str, events: dict, archival: dict) -> list[str]:
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

    return _sort_phases(phases)


def get_phases_for_branch_accumulated(events: list[dict], archival: list[dict]) -> list[str]:
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

    return _sort_phases(phases)


def _sort_phases(phases: set[str]) -> list[str]:
    """Sort phases in canonical order."""
    phase_order = ["phase0", "phase1", "phase2", "phase3", "phase4", "summary"]
    sorted_phases = []
    for p in phase_order:
        if p in phases:
            sorted_phases.append(p)
    for p in sorted(phases):
        if p not in sorted_phases:
            sorted_phases.append(p)
    return sorted_phases


def escape_html(text: str) -> str:
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


def get_ancestor_chain(branch_id: str, branches: list[dict]) -> list[dict]:
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


def collect_inherited_data(
    branch_id: str,
    branches: list[dict],
    core_kv: dict,
    events: dict,
    archival: dict,
    branch_to_index: dict,
    exclude_virtual_root: bool = False,
    own_core_keys: set[str] | None = None,
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
        own_core_keys: Set of keys that exist in this node's own core_kv.
                       These keys are excluded from inherited data since they've been overridden.

    Returns:
        Tuple of (inherited_core_kv, inherited_events, inherited_archival, ancestors_info)
    """
    ancestors = get_ancestor_chain(branch_id, branches)

    inherited_events = []
    inherited_archival = []
    ancestors_info = []

    # For core_kv: collect all values first, then select the most recent for each key
    # This matches the actual memory system behavior (_render_core_memory)
    core_kv_candidates: dict[str, list[dict]] = {}

    # Keys that are overridden by this node should not appear in inherited data
    excluded_keys = own_core_keys or set()

    # Collect data from each ancestor (root to parent order)
    for ancestor in ancestors:
        ancestor_id = ancestor["id"]
        ancestor_node_uid = ancestor.get("node_uid")
        ancestor_index = branch_to_index.get(ancestor_id, -1)

        # Only "none_root" is truly virtual (added by tree viz for orphan nodes)
        # "root" is the actual root node with data, not a virtual placeholder
        is_virtual = ancestor_node_uid == "none_root"

        ancestors_info.append({
            "index": ancestor_index,
            "branch_id": ancestor_id,
            "node_uid": ancestor_node_uid or ancestor_id,
            "is_virtual": is_virtual,
        })

        # Skip collecting data from virtual root (only none_root, not actual root)
        if exclude_virtual_root and is_virtual:
            continue

        # Collect core_kv candidates (we'll select the most recent later)
        for kv in core_kv.get(ancestor_id, []):
            key = kv.get("key")
            if key and key not in excluded_keys:
                if key not in core_kv_candidates:
                    core_kv_candidates[key] = []
                core_kv_candidates[key].append(kv)

        # Collect events
        for event in events.get(ancestor_id, []):
            inherited_events.append(event)

        # Collect archival
        for arch in archival.get(ancestor_id, []):
            inherited_archival.append(arch)

    # Select the most recent value for each key (matching actual memory system behavior)
    inherited_core_kv = []
    for key, candidates in core_kv_candidates.items():
        # Sort by updated_at descending and take the most recent
        candidates.sort(key=lambda kv: kv.get("updated_at") or "", reverse=True)
        inherited_core_kv.append(candidates[0])

    return inherited_core_kv, inherited_events, inherited_archival, ancestors_info


def load_journal_nodes_from_tree_data(log_dir: Path) -> list[dict]:
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


def merge_journal_nodes_into_memory_data(memory_data: dict, journal_nodes: list[dict]) -> dict:
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


def index_memory_calls_by_branch(
    memory_calls: list[dict],
    branch_ids: set[str],
) -> dict[str, list[dict]]:
    """Index memory call log events by branch_id.

    Args:
        memory_calls: List of memory call events from memory_calls.jsonl
        branch_ids: Set of valid branch IDs to index

    Returns:
        Dictionary mapping branch_id to list of memory call events
    """
    indexed: dict[str, list[dict]] = defaultdict(list)

    for event in memory_calls:
        if not isinstance(event, dict):
            continue
        branch_id = event.get("branch_id")
        if branch_id and branch_id in branch_ids:
            # Set default phase for mem_node_fork operations if phase is null
            if event.get("op") == "mem_node_fork" and event.get("phase") is None:
                event = dict(event)  # Make a copy to avoid modifying original
                event["phase"] = "tree_structure"
            indexed[branch_id].append(event)

    # Sort each branch's events by timestamp
    for branch_id in indexed:
        indexed[branch_id].sort(key=lambda e: e.get("ts", 0) or 0)

    return dict(indexed)


def _load_template_asset(asset_path: Path) -> str:
    """Load a template asset file (CSS or JS).

    Args:
        asset_path: Path to the asset file

    Returns:
        Content of the asset file as string
    """
    try:
        with open(asset_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[yellow]Warning: Could not load asset {asset_path}: {e}[/yellow]")
        return ""


def generate_memory_database_html(
    memory_data: dict,
    output_path: Path,
    experiment_name: str = "Memory Database",
    memory_calls: list[dict] | None = None,
):
    """Generate the HTML visualization for memory database.

    Uses a modular template system where CSS and JS assets are inlined
    from separate files for better maintainability.

    Args:
        memory_data: Dictionary with branches, core_kv, events, archival
        output_path: Path to write the HTML file
        experiment_name: Name to display in the visualization
        memory_calls: Optional list of memory call log events (from memory_calls.jsonl)
    """
    branches = memory_data["branches"]
    layout, edges = build_memory_tree_layout(branches)

    # Build branch_id to index mapping
    branch_to_index = {b["id"]: i for i, b in enumerate(branches)}
    branch_ids = set(branch_to_index.keys())

    # Index memory calls by branch if provided
    memory_calls_by_branch: dict[str, list[dict]] = {}
    if memory_calls:
        memory_calls_by_branch = index_memory_calls_by_branch(memory_calls, branch_ids)

    nodes_data = []
    for i, branch in enumerate(branches):
        branch_id = branch["id"]
        node_uid = branch.get("node_uid")
        # Only "none_root" is truly virtual (artificial placeholder for orphan nodes)
        # "root" is the actual root node with real data
        is_virtual_node = node_uid == "none_root"

        if is_virtual_node:
            # Virtual node (none_root placeholder) should have no memory data
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
                "own_memory_calls": [],
                "inherited_core_kv": [],
                "inherited_events": [],
                "inherited_archival": [],
                "inherited_memory_calls": [],
                "ancestors": [],
                # Legacy fields
                "core_kv": [],
                "events": [],
                "archival": [],
                "memory_calls": [],
            })
            continue

        # Get own data (this branch only)
        own_core_kv = memory_data["core_kv"].get(branch_id, [])
        own_events = memory_data["events"].get(branch_id, [])
        own_archival = memory_data["archival"].get(branch_id, [])
        own_memory_calls = memory_calls_by_branch.get(branch_id, [])

        # Extract keys from own_core_kv to exclude from inherited data
        # (keys that are overridden by this node should not appear in inherited)
        own_core_keys = {kv.get("key") for kv in own_core_kv if kv.get("key")}

        # Get inherited data from ancestors (excluding virtual root's data)
        inherited_core_kv, inherited_events, inherited_archival, ancestors_info = collect_inherited_data(
            branch_id,
            branches,
            memory_data["core_kv"],
            memory_data["events"],
            memory_data["archival"],
            branch_to_index,
            exclude_virtual_root=True,
            own_core_keys=own_core_keys,
        )

        # Collect inherited memory calls from ancestors
        inherited_memory_calls = []
        for ancestor in ancestors_info:
            if ancestor.get("is_virtual"):
                continue
            ancestor_branch_id = ancestor.get("branch_id")
            if ancestor_branch_id:
                inherited_memory_calls.extend(memory_calls_by_branch.get(ancestor_branch_id, []))

        # Get inherited exclusions and summaries for this branch (Copy-on-Write data)
        branch_exclusions = set(memory_data.get("inherited_exclusions", {}).get(branch_id, []))
        branch_summaries = memory_data.get("inherited_summaries", {}).get(branch_id, [])

        # Filter inherited events based on exclusions (Copy-on-Write semantics)
        # Events that have been consolidated are excluded from inherited view
        filtered_inherited_events = [
            event for event in inherited_events
            if event.get("id") not in branch_exclusions
        ]

        # Combine own and inherited for phase detection
        all_events = own_events + filtered_inherited_events
        all_archival = own_archival + inherited_archival
        phases = get_phases_for_branch_accumulated(all_events, all_archival)

        # Build effective memory (what LLM actually sees)
        # For core_kv: own values override inherited (matching actual memory system)
        effective_core_kv = list(own_core_kv)  # Start with own
        inherited_keys = own_core_keys  # Keys we already have from own
        for kv in inherited_core_kv:
            if kv.get("key") not in inherited_keys:
                effective_core_kv.append(kv)
        # Sort by key for consistency
        effective_core_kv.sort(key=lambda kv: kv.get("key", ""))

        # For events: combine own + filtered inherited + inherited summaries (sorted by created_at)
        # Inherited summaries represent consolidated ancestor events
        effective_events = sorted(
            all_events,
            key=lambda e: e.get("created_at") or "",
        )
        effective_archival = sorted(
            all_archival,
            key=lambda a: a.get("created_at") or "",
        )

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
            "own_memory_calls": own_memory_calls,
            # Inherited data (from ancestors, with Copy-on-Write exclusions applied)
            "inherited_core_kv": inherited_core_kv,
            "inherited_events": filtered_inherited_events,  # Filtered by exclusions
            "inherited_archival": inherited_archival,
            "inherited_memory_calls": inherited_memory_calls,
            # Inherited memory consolidation (Copy-on-Write)
            "inherited_exclusions": list(branch_exclusions),
            "inherited_summaries": branch_summaries,
            # Effective memory (what LLM actually sees = own + inherited merged)
            "effective_core_kv": effective_core_kv,
            "effective_events": effective_events,
            "effective_archival": effective_archival,
            # Ancestor information
            "ancestors": ancestors_info,
            # Legacy fields for backwards compatibility
            "core_kv": own_core_kv,
            "events": own_events,
            "archival": own_archival,
            "memory_calls": own_memory_calls,
        })

    js_data = {
        "layout": layout,
        "edges": edges,
        "nodes": nodes_data,
    }

    # Load modular template and assets
    template_dir = Path(__file__).parent / "templates"
    assets_dir = template_dir / "assets"

    # Try to use new modular template (v2), fall back to legacy if not found
    template_v2_path = template_dir / "memory_database_v2.html"
    if template_v2_path.exists():
        # Load the v2 template with asset placeholders
        with open(template_v2_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Load and inline CSS assets
        common_css = _load_template_asset(assets_dir / "common.css")
        memory_database_css = _load_template_asset(assets_dir / "memory_database.css")

        # Load and inline JS assets
        resizable_js = _load_template_asset(assets_dir / "resizable.js")
        memory_database_js = _load_template_asset(assets_dir / "memory_database.js")
        tree_canvas_js = _load_template_asset(assets_dir / "tree_canvas.js")

        # Replace CSS placeholders
        html_content = html_template.replace("__COMMON_CSS__", common_css)
        html_content = html_content.replace("__MEMORY_DATABASE_CSS__", memory_database_css)

        # Replace JS placeholders
        html_content = html_content.replace("__RESIZABLE_JS__", resizable_js)
        html_content = html_content.replace("__MEMORY_DATABASE_JS__", memory_database_js)
        html_content = html_content.replace("__TREE_CANVAS_JS__", tree_canvas_js)
    else:
        # Fall back to legacy template
        legacy_template_path = template_dir / "memory_database.html"
        with open(legacy_template_path, "r", encoding="utf-8") as f:
            html_content = f.read()

    # Replace data placeholders
    html_content = html_content.replace("__EXPERIMENT_NAME__", escape_html(experiment_name))
    html_content = html_content.replace("__JS_DATA__", json.dumps(js_data, ensure_ascii=False))

    output_path.write_text(html_content, encoding='utf-8')


def create_memory_database_viz(cfg: Any, current_stage_viz_path: Path):
    """
    Create a memory database visualization HTML file.
    This will be placed in the main log directory alongside unified_tree_viz.html.

    Args:
        cfg: Configuration object with workspace_dir attribute
        current_stage_viz_path: Path to the current stage visualization file
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
        memory_data = load_memory_database(db_path)

        # Load journal nodes from tree_data.json files and merge with SQLite data
        journal_nodes = load_journal_nodes_from_tree_data(log_dir)
        if journal_nodes:
            memory_data = merge_journal_nodes_into_memory_data(memory_data, journal_nodes)
            print(f"[blue]Merged {len(journal_nodes)} journal nodes into memory data[/blue]")

        # Load memory call log (includes memory reads/injections)
        # Check both workspace_dir and log_dir for memory_calls.jsonl
        memory_calls = load_memory_call_log(workspace_dir)
        if not memory_calls:
            memory_calls = load_memory_call_log(log_dir)

        if memory_calls:
            print(f"[blue]Loaded {len(memory_calls)} memory call log entries[/blue]")
        else:
            print(f"[yellow]No memory_calls.jsonl found, memory reads will not be displayed[/yellow]")

        output_path = log_dir / "memory_database.html"
        experiment_name = log_dir.name
        generate_memory_database_html(memory_data, output_path, experiment_name, memory_calls)
        print(f"[green]Created memory database visualization at {output_path}[/green]")
    except Exception as e:
        print(f"[red]Error generating memory database visualization: {e}[/red]")
