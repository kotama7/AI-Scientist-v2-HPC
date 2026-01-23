"""Run I/O utilities for saving experiment results.

This module is intentionally separated from config.py to avoid circular imports.
It can safely import from both config and journal modules.
"""

from pathlib import Path

from omegaconf import OmegaConf

from . import serialize, tree_export
from .config import Config
from ..journal import Journal


def save_run(cfg: Config, journal: Journal, stage_name: str = None):
    """Save the experiment run results to disk.

    Args:
        cfg: Configuration object.
        journal: Journal containing experiment nodes.
        stage_name: Name of the stage (used for directory naming).
    """
    if stage_name is None:
        stage_name = "NoStageRun"
    save_dir = cfg.log_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    try:
        serialize.dump_json(journal, save_dir / "journal.json")
    except Exception as e:
        print(f"Error saving journal: {e}")
        raise
    # save config
    try:
        OmegaConf.save(config=cfg, f=save_dir / "config.yaml")
    except Exception as e:
        print(f"Error saving config: {e}")
        raise
    # create the tree + code visualization
    try:
        tree_export.generate(cfg, journal, save_dir / "tree_plot.html")
    except Exception as e:
        print(f"Error generating tree: {e}")
        raise
    # save the best found solution
    try:
        best_node = journal.get_best_node(only_good=False, cfg=cfg)
        if best_node is not None:
            suffix = Path(cfg.exec.agent_file_name).suffix or ".py"
            if getattr(cfg.exec, "phase_mode", "single") == "split":
                suffix = ".txt"
            for existing_file in save_dir.glob(f"best_solution_*.{suffix.lstrip('.')}"):
                existing_file.unlink()
            # Create new best solution file
            filename = f"best_solution_{best_node.id}{suffix}"
            with open(save_dir / filename, "w") as f:
                f.write(best_node.code)
            # save best_node.id to a text file
            with open(save_dir / "best_node_id.txt", "w") as f:
                f.write(str(best_node.id))
        else:
            print("No best node found yet")
    except Exception as e:
        print(f"Error saving best solution: {e}")
