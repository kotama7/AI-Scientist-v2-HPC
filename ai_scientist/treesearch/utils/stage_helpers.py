"""Stage directory helper utilities."""

from __future__ import annotations

from pathlib import Path


def get_completed_stages(log_dir: Path) -> list[str]:
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


def get_substage_dir_map(log_dir: Path) -> dict[str, list[str]]:
    """Return all sub-stage directories for each main stage, sorted by sub-stage number.

    Returns a mapping like ``{"Stage_3": ["stage_3_..._1_first", "stage_3_..._2_next"]}``.
    Only stages with more than one sub-stage directory (that contain tree_data.json)
    are included.
    """
    import re

    # Collect dirs per main stage
    stage_dirs: dict[str, list[tuple[int, str]]] = {}
    for d in log_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("stage_"):
            continue
        parts = d.name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        # Must contain tree_data.json to be valid
        if not (d / "tree_data.json").exists():
            continue
        stage_id = f"Stage_{parts[1]}"
        # Extract sub-stage number from the dir name pattern: stage_N_name_SUBNUM_subname
        nums = [int(n) for n in re.findall(r"\d+", d.name)]
        sub_num = nums[1] if len(nums) >= 2 else 1
        stage_dirs.setdefault(stage_id, []).append((sub_num, d.name))

    # Only return stages with multiple sub-stage dirs
    result: dict[str, list[str]] = {}
    for stage_id, dirs in stage_dirs.items():
        if len(dirs) > 1:
            dirs.sort(key=lambda x: x[0])
            result[stage_id] = [name for _, name in dirs]
    return result


def stage_dir_to_stage_id(stage_dir_name: str) -> str | None:
    """Convert a stage directory name (e.g., 'stage_1_foo') to stage ID (e.g., 'Stage_1')."""
    if not stage_dir_name.startswith("stage_"):
        return None
    parts = stage_dir_name.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return f"Stage_{parts[1]}"
