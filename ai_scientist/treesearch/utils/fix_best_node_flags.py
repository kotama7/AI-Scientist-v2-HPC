"""Fix is_best_node flags in tree_data.json files based on actual inheritance.

This utility fixes the inconsistency where best_node_id.txt may differ from
the node actually inherited to the next stage (recorded in inherited_from_node_id).

The authoritative source of truth is the inherited_from_node_id in the next stage's
tree_data.json, as it records which node was actually selected by
_get_best_implementation() at stage transition time.
"""

import json
from pathlib import Path
from rich import print


def fix_best_node_flags_in_experiment(log_dir: Path, dry_run: bool = False) -> dict[str, dict]:
    """Fix is_best_node flags in all tree_data.json files in an experiment.

    Args:
        log_dir: Path to the experiment log directory (e.g., .../logs/0-run/)
        dry_run: If True, only report what would be changed without modifying files

    Returns:
        Dictionary mapping stage names to fix information
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise ValueError(f"Log directory does not exist: {log_dir}")

    # Find all stage directories with tree_data.json
    stage_dirs = []
    for d in sorted(log_dir.iterdir()):
        if d.is_dir() and d.name.startswith("stage_"):
            tree_data_path = d / "tree_data.json"
            if tree_data_path.exists():
                stage_dirs.append(d)

    if not stage_dirs:
        print(f"[yellow]No stage directories with tree_data.json found in {log_dir}[/yellow]")
        return {}

    print(f"[blue]Found {len(stage_dirs)} stages with tree_data.json[/blue]")

    # Build a map of which node was inherited to each stage
    inherited_map = {}  # stage_dir_name -> (inherited_node_id, source_stage_dir_name)

    for i, stage_dir in enumerate(stage_dirs):
        with open(stage_dir / "tree_data.json") as f:
            data = json.load(f)

        inherited_ids = data.get("inherited_from_node_id", [])
        if inherited_ids and inherited_ids[0]:
            inherited_id = inherited_ids[0]
            # The inherited node came from the previous stage
            if i > 0:
                prev_stage_dir = stage_dirs[i - 1]
                inherited_map[stage_dir.name] = (inherited_id, prev_stage_dir.name)
                print(f"[cyan]  {stage_dir.name} inherited {inherited_id} from {prev_stage_dir.name}[/cyan]")

    # Now fix is_best_node flags in each stage based on what was inherited to the next stage
    fixes = {}

    for i, stage_dir in enumerate(stage_dirs):
        stage_name = stage_dir.name

        # Check if the next stage inherited a node from this stage
        next_stage_inherited = None
        if i + 1 < len(stage_dirs):
            next_stage_name = stage_dirs[i + 1].name
            if next_stage_name in inherited_map:
                inherited_id, source_stage = inherited_map[next_stage_name]
                if source_stage == stage_name:
                    next_stage_inherited = inherited_id

        # Load current tree_data.json
        tree_data_path = stage_dir / "tree_data.json"
        with open(tree_data_path) as f:
            data = json.load(f)

        node_ids = data["node_id"]
        is_best_node = data["is_best_node"]

        # Find current best node
        current_best_indices = [idx for idx, is_best in enumerate(is_best_node) if is_best]
        current_best_id = node_ids[current_best_indices[0]] if current_best_indices else None

        # Find what should be the best node
        correct_best_id = None
        source_of_truth = None

        if next_stage_inherited:
            # Next stage's inherited_from_node_id is the source of truth
            correct_best_id = next_stage_inherited
            source_of_truth = "next_stage_inherited_from_node_id"
        else:
            # No next stage or no inheritance - check best_node_id.txt
            best_node_file = stage_dir / "best_node_id.txt"
            if best_node_file.exists():
                correct_best_id = best_node_file.read_text().strip()
                source_of_truth = "best_node_id.txt"

        # Check if fix is needed
        needs_fix = False
        if correct_best_id and correct_best_id != current_best_id:
            needs_fix = True

        fix_info = {
            "stage_name": stage_name,
            "current_best_id": current_best_id,
            "current_best_index": current_best_indices[0] if current_best_indices else None,
            "correct_best_id": correct_best_id,
            "correct_best_index": node_ids.index(correct_best_id) if correct_best_id and correct_best_id in node_ids else None,
            "source_of_truth": source_of_truth,
            "needs_fix": needs_fix,
        }

        fixes[stage_name] = fix_info

        if needs_fix:
            print(f"\n[yellow]Stage: {stage_name}[/yellow]")
            print(f"  Current best: {current_best_id} (index {fix_info['current_best_index']})")
            print(f"  Correct best: {correct_best_id} (index {fix_info['correct_best_index']})")
            print(f"  Source: {source_of_truth}")

            if not dry_run:
                # Update is_best_node flags
                new_is_best_node = [False] * len(is_best_node)
                if fix_info['correct_best_index'] is not None:
                    new_is_best_node[fix_info['correct_best_index']] = True

                data["is_best_node"] = new_is_best_node

                # Write updated tree_data.json
                with open(tree_data_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"  [green]✓ Fixed is_best_node flags in {tree_data_path}[/green]")

                # Also update best_node_id.txt to match
                best_node_file = stage_dir / "best_node_id.txt"
                with open(best_node_file, "w") as f:
                    f.write(correct_best_id)
                print(f"  [green]✓ Updated {best_node_file}[/green]")
            else:
                print(f"  [blue](dry run - no changes made)[/blue]")
        else:
            if correct_best_id:
                print(f"[green]✓ {stage_name}: is_best_node already correct ({current_best_id})[/green]")

    return fixes


def main():
    """Command-line interface for fixing best node flags."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix is_best_node flags in tree_data.json based on actual inheritance"
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path to experiment log directory (e.g., experiments/.../logs/0-run/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    print(f"\n[bold]Fixing best_node flags in {args.log_dir}[/bold]\n")
    fixes = fix_best_node_flags_in_experiment(args.log_dir, dry_run=args.dry_run)

    # Summary
    num_fixed = sum(1 for fix in fixes.values() if fix["needs_fix"])
    num_correct = len(fixes) - num_fixed

    print(f"\n[bold]Summary:[/bold]")
    print(f"  Total stages: {len(fixes)}")
    print(f"  Already correct: {num_correct}")
    print(f"  Fixed: {num_fixed}")

    if args.dry_run and num_fixed > 0:
        print(f"\n[yellow]Dry run mode - no changes were made.[/yellow]")
        print(f"[yellow]Run without --dry-run to apply fixes.[/yellow]")


if __name__ == "__main__":
    main()
