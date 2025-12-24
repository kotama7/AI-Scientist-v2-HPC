"""
Resource configuration for external data, GitHub repos, and Hugging Face models/datasets.

Supports three resource types:
- Local: Bind-mounted directories from host (read-only)
- GitHub: Repos cloned inside container with fixed ref (commit/tag)
- HuggingFace: Models/datasets downloaded via huggingface_hub
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

logger = logging.getLogger("ai-scientist")


@dataclass
class LocalResource:
    """Local directory to bind-mount into container."""

    name: str
    host_path: str
    mount_path: str
    read_only: bool = True

    def validate(self) -> list[str]:
        """Return validation errors if any."""
        errors: list[str] = []
        if not self.name:
            errors.append("LocalResource.name is required")
        if not self.host_path:
            errors.append(f"LocalResource '{self.name}': host_path is required")
        elif not Path(self.host_path).exists():
            errors.append(f"LocalResource '{self.name}': host_path does not exist: {self.host_path}")
        if not self.mount_path:
            errors.append(f"LocalResource '{self.name}': mount_path is required")
        elif not self.mount_path.startswith("/workspace"):
            errors.append(f"LocalResource '{self.name}': mount_path must be under /workspace: {self.mount_path}")
        return errors


@dataclass
class GitHubResource:
    """GitHub repository to clone inside container."""

    name: str
    repo: str
    dest: str
    ref: str | None = None
    as_: Literal["library", "data"] = "data"

    def validate(self) -> list[str]:
        """Return validation errors if any."""
        errors: list[str] = []
        if not self.name:
            errors.append("GitHubResource.name is required")
        if not self.repo:
            errors.append(f"GitHubResource '{self.name}': repo is required")
        if not self.dest:
            errors.append(f"GitHubResource '{self.name}': dest is required")
        elif not self.dest.startswith("/workspace"):
            errors.append(f"GitHubResource '{self.name}': dest must be under /workspace: {self.dest}")
        if not self.ref:
            logger.warning("GitHubResource '%s': ref not specified; recommend using fixed commit SHA or tag", self.name)
        return errors


@dataclass
class HuggingFaceResource:
    """Hugging Face model or dataset to download inside container."""

    name: str
    type: Literal["model", "dataset"]
    repo_id: str
    dest: str
    revision: str | None = None

    def validate(self) -> list[str]:
        """Return validation errors if any."""
        errors: list[str] = []
        if not self.name:
            errors.append("HuggingFaceResource.name is required")
        if not self.repo_id:
            errors.append(f"HuggingFaceResource '{self.name}': repo_id is required")
        if not self.dest:
            errors.append(f"HuggingFaceResource '{self.name}': dest is required")
        elif not self.dest.startswith("/workspace"):
            errors.append(f"HuggingFaceResource '{self.name}': dest must be under /workspace: {self.dest}")
        if self.type not in ("model", "dataset"):
            errors.append(f"HuggingFaceResource '{self.name}': type must be 'model' or 'dataset'")
        if not self.revision:
            logger.warning("HuggingFaceResource '%s': revision not specified; recommend using fixed commit SHA", self.name)
        return errors


@dataclass
class ResourceConfig:
    """Container for all resource definitions."""

    local: list[LocalResource] = field(default_factory=list)
    github: list[GitHubResource] = field(default_factory=list)
    huggingface: list[HuggingFaceResource] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate all resources and return errors."""
        errors: list[str] = []
        for res in self.local:
            errors.extend(res.validate())
        for res in self.github:
            errors.extend(res.validate())
        for res in self.huggingface:
            errors.extend(res.validate())
        return errors

    def has_resources(self) -> bool:
        """Check if any resources are defined."""
        return bool(self.local or self.github or self.huggingface)


def _parse_local_resource(data: dict[str, Any]) -> LocalResource:
    return LocalResource(
        name=data.get("name", ""),
        host_path=data.get("host_path", ""),
        mount_path=data.get("mount_path", ""),
        read_only=data.get("read_only", True),
    )


def _parse_github_resource(data: dict[str, Any]) -> GitHubResource:
    return GitHubResource(
        name=data.get("name", ""),
        repo=data.get("repo", ""),
        ref=data.get("ref"),
        as_=data.get("as", "data"),
        dest=data.get("dest", ""),
    )


def _parse_huggingface_resource(data: dict[str, Any]) -> HuggingFaceResource:
    return HuggingFaceResource(
        name=data.get("name", ""),
        type=data.get("type", "model"),
        repo_id=data.get("repo_id", ""),
        revision=data.get("revision"),
        dest=data.get("dest", ""),
    )


def load_resources(path: str | Path) -> ResourceConfig:
    """
    Load resource configuration from JSON or YAML file.

    Args:
        path: Path to the resources file (.json or .yaml/.yml)

    Returns:
        ResourceConfig with parsed resources

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If validation fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resources file not found: {path}")

    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(content)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML resource files: pip install pyyaml")
        data = yaml.safe_load(content)
    else:
        # Try JSON first, then YAML
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml
                data = yaml.safe_load(content)
            except Exception:
                raise ValueError(f"Cannot parse resources file as JSON or YAML: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Resources file must contain a JSON/YAML object: {path}")

    config = ResourceConfig(
        local=[_parse_local_resource(r) for r in data.get("local", [])],
        github=[_parse_github_resource(r) for r in data.get("github", [])],
        huggingface=[_parse_huggingface_resource(r) for r in data.get("huggingface", [])],
    )

    errors = config.validate()
    if errors:
        raise ValueError(f"Resource validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config


def build_local_binds(resources: ResourceConfig | None) -> list[str]:
    """
    Build Singularity --bind arguments for local resources.

    Returns list of bind specs in format "host_path:mount_path:ro" or "host_path:mount_path"
    """
    if not resources:
        return []

    binds: list[str] = []
    for res in resources.local:
        spec = f"{res.host_path}:{res.mount_path}"
        if res.read_only:
            spec += ":ro"
        binds.append(spec)
    return binds


def get_github_fetch_commands(resources: ResourceConfig | None) -> list[dict[str, Any]]:
    """
    Generate fetch commands for GitHub resources.

    Returns list of dicts with 'command' and 'resource' for tracking.
    """
    if not resources:
        return []

    commands: list[dict[str, Any]] = []
    for res in resources.github:
        # Use shallow clone when possible, but need full clone for commit SHA
        if res.ref and len(res.ref) == 40:  # Full SHA
            # Need to fetch, then checkout
            cmd = (
                f"git clone --no-checkout {res.repo} {res.dest} && "
                f"cd {res.dest} && "
                f"git fetch origin {res.ref} && "
                f"git checkout {res.ref}"
            )
        elif res.ref:
            # Tag or branch - can use shallow clone
            cmd = f"git clone --depth 1 --branch {res.ref} {res.repo} {res.dest}"
        else:
            # No ref - clone default branch
            cmd = f"git clone --depth 1 {res.repo} {res.dest}"

        commands.append({
            "command": cmd,
            "resource": {
                "type": "github",
                "name": res.name,
                "repo": res.repo,
                "ref": res.ref,
                "dest": res.dest,
                "as": res.as_,
            },
        })

        # Add verification command to log commit SHA
        verify_cmd = f"cd {res.dest} && git rev-parse HEAD"
        commands.append({
            "command": verify_cmd,
            "resource": {
                "type": "github_verify",
                "name": res.name,
            },
        })

    return commands


def get_huggingface_fetch_commands(resources: ResourceConfig | None) -> list[dict[str, Any]]:
    """
    Generate fetch commands for Hugging Face resources.

    Returns list of dicts with 'command' and 'resource' for tracking.
    """
    if not resources:
        return []

    commands: list[dict[str, Any]] = []
    for res in resources.huggingface:
        # Build Python command for snapshot download
        repo_type = "dataset" if res.type == "dataset" else "model"
        revision_arg = f', revision="{res.revision}"' if res.revision else ""

        python_code = (
            f"from huggingface_hub import snapshot_download; "
            f'path = snapshot_download("{res.repo_id}", local_dir="{res.dest}", '
            f'repo_type="{repo_type}"{revision_arg}); '
            f"print(f'Downloaded to: {{path}}')"
        )
        cmd = f"HF_HOME=/workspace/.cache/huggingface python -c '{python_code}'"

        commands.append({
            "command": cmd,
            "resource": {
                "type": "huggingface",
                "name": res.name,
                "repo_id": res.repo_id,
                "hf_type": res.type,
                "revision": res.revision,
                "dest": res.dest,
            },
        })

    return commands


def get_all_fetch_commands(resources: ResourceConfig | None) -> list[dict[str, Any]]:
    """Get all fetch commands for GitHub and HuggingFace resources."""
    commands: list[dict[str, Any]] = []
    commands.extend(get_github_fetch_commands(resources))
    commands.extend(get_huggingface_fetch_commands(resources))
    return commands


def build_resources_context(resources: ResourceConfig | None) -> dict[str, Any]:
    """
    Build context dict for prompt injection.

    Returns structured information about resources for LLM context.
    """
    if not resources:
        return {"has_resources": False}

    local_mounts: list[dict[str, str]] = []
    for res in resources.local:
        local_mounts.append({
            "name": res.name,
            "path": res.mount_path,
            "read_only": res.read_only,
        })

    github_resources: list[dict[str, Any]] = []
    for res in resources.github:
        github_resources.append({
            "name": res.name,
            "repo": res.repo,
            "ref": res.ref or "(latest)",
            "dest": res.dest,
            "as": res.as_,
        })

    hf_resources: list[dict[str, Any]] = []
    for res in resources.huggingface:
        hf_resources.append({
            "name": res.name,
            "type": res.type,
            "repo_id": res.repo_id,
            "revision": res.revision or "(latest)",
            "dest": res.dest,
        })

    return {
        "has_resources": resources.has_resources(),
        "local_mounts": local_mounts,
        "github_resources": github_resources,
        "huggingface_resources": hf_resources,
        "hf_home": "/workspace/.cache/huggingface",
        "install_paths": {
            "cmake": "/workspace/.local",
            "python": "/workspace/.pydeps",
        },
    }
