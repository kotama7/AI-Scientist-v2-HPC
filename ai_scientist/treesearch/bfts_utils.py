import os
import os.path as osp
import shutil
from typing import Sequence

import yaml


def idea_to_markdown(
    data: dict, output_path: str, load_code: str | None = None, code_fence: str = "python"
) -> None:
    """
    Convert a dictionary into a markdown file.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Optional path to a code file to include in the markdown
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # Convert key to title format and make it a header
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            # Handle different value types
            if isinstance(value, (list, tuple)):
                for item in value:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n")
                    f.write(f"{sub_value}\n\n")
            else:
                f.write(f"{value}\n\n")

        # Add the code to the markdown file
        if load_code:
            # Assert that the code file exists before trying to open it
            assert os.path.exists(load_code), (
                f"Code path at {load_code} must exist when load_code is provided."
            )
            f.write(f"## Code To Potentially Use\n\n")
            f.write(f"Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```{code_fence}\n{code}\n```\n\n")


def edit_bfts_config_file(
    config_path: str,
    idea_dir: str,
    idea_path: str,
    *,
    language: str | None = None,
    agent_file_name: str | None = None,
    env_packages_template: str | None = None,
    phase_mode: str | None = None,
    singularity_image: str | None = None,
    use_gpu: bool | None = None,
    num_workers: int | None = None,
    writable_tmpfs: bool | None = None,
    container_overlay: str | None = None,
    container_extra_args: Sequence[str] | None = None,
    per_worker_sif: bool | None = None,
    keep_sandbox: bool | None = None,
    use_fakeroot: bool | None = None,
    writable_mode: str | None = None,
    phase1_max_steps: int | None = None,
    resources_path: str | None = None,
    memory_enabled: bool | None = None,
    memory_db_path: str | None = None,
    memory_core_max_chars: int | None = None,
    memory_recall_max_events: int | None = None,
    memory_retrieval_k: int | None = None,
) -> str:
    """
    Edit the bfts_config.yaml file to point to the idea.md file

    Args:
        config_path: Path to the bfts_config.yaml file
        idea_dir: Directory where the idea.md file is located
        idea_path: Path to the idea.md file

    Returns:
        Path to the edited bfts_config.yaml file
    """
    run_config_path = osp.join(idea_dir, "bfts_config.yaml")
    shutil.copy(config_path, run_config_path)
    with open(run_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["desc_file"] = idea_path
    config["workspace_dir"] = idea_dir

    # make an empty data directory
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = data_dir

    # make an empty log directory
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    exec_cfg = config.setdefault("exec", {})
    if language is not None:
        exec_cfg["language"] = language
    if agent_file_name is not None:
        exec_cfg["agent_file_name"] = agent_file_name
    if env_packages_template is not None:
        exec_cfg["env_packages_template"] = env_packages_template
    if phase_mode is not None:
        exec_cfg["phase_mode"] = phase_mode
    if singularity_image is not None:
        exec_cfg["singularity_image"] = singularity_image
    if use_gpu is not None:
        exec_cfg["use_gpu"] = bool(use_gpu)
    if writable_tmpfs is not None:
        exec_cfg["writable_tmpfs"] = bool(writable_tmpfs)
    if container_overlay is not None:
        exec_cfg["container_overlay"] = container_overlay
    if container_extra_args is not None:
        exec_cfg["container_extra_args"] = list(container_extra_args)
    if per_worker_sif is not None:
        exec_cfg["per_worker_sif"] = bool(per_worker_sif)
    if keep_sandbox is not None:
        exec_cfg["keep_sandbox"] = bool(keep_sandbox)
    if use_fakeroot is not None:
        exec_cfg["use_fakeroot"] = bool(use_fakeroot)
    if writable_mode is not None:
        exec_cfg["writable_mode"] = writable_mode
    if phase1_max_steps is not None:
        exec_cfg["phase1_max_steps"] = int(phase1_max_steps)
    if resources_path is not None:
        exec_cfg["resources"] = resources_path
    if num_workers is not None:
        config.setdefault("agent", {})
        config["agent"]["num_workers"] = int(num_workers)

    mem_cfg = config.setdefault("memory", {})
    if memory_enabled is not None:
        mem_cfg["enabled"] = bool(memory_enabled)
    if memory_db_path is not None:
        mem_cfg["db_path"] = memory_db_path
    if memory_core_max_chars is not None:
        mem_cfg["core_max_chars"] = int(memory_core_max_chars)
    if memory_recall_max_events is not None:
        mem_cfg["recall_max_events"] = int(memory_recall_max_events)
    if memory_retrieval_k is not None:
        mem_cfg["retrieval_k"] = int(memory_retrieval_k)

    with open(run_config_path, "w") as f:
        yaml.dump(config, f)
    return run_config_path
