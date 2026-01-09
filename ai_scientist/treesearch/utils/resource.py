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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from .phase_execution import summarize_text

logger = logging.getLogger("ai-scientist")

ResourceClass = Literal["template", "library", "dataset", "model", "setup", "document"]
ResourceSource = Literal["local", "github", "huggingface"]

RESOURCE_CLASS_GUIDANCE: dict[str, list[str]] = {
    "template": [
        "Use as reference code; preserve the original and copy into workspace if you need changes.",
        "Follow the original file structure and implementation patterns when designing experiments.",
        "Prefer citing paths from the template when deriving or adapting code.",
    ],
    "library": [
        "Treat as a dependency; avoid editing unless explicitly needed.",
        "Install/build under /workspace/.local (C/C++) or /workspace/.pydeps (Python).",
    ],
    "dataset": [
        "Read-only data; do not modify in place.",
        "Record the container path and use it for evaluation or preprocessing.",
    ],
    "model": [
        "Use for inference or initialization; do not retrain unless required.",
        "Keep large model artifacts out of the final output unless requested.",
    ],
    "setup": [
        "Run setup scripts or apply configs in Phase 1, then verify outputs.",
        "Keep setup artifacts under /workspace.",
    ],
    "document": [
        "Use as guidance and cite key constraints or assumptions.",
        "Prefer summarizing documents rather than copying large blocks verbatim.",
    ],
}

RESOURCE_CLASS_DEFAULTS: dict[str, dict[str, int | bool]] = {
    "template": {"include_tree": True, "include_content": True, "max_files": 12, "max_chars": 3500, "max_total_chars": 12000},
    "library": {"include_tree": False, "include_content": False, "max_files": 6, "max_chars": 2000, "max_total_chars": 4000},
    "dataset": {"include_tree": False, "include_content": False, "max_files": 0, "max_chars": 0, "max_total_chars": 0},
    "model": {"include_tree": False, "include_content": False, "max_files": 0, "max_chars": 0, "max_total_chars": 0},
    "setup": {"include_tree": True, "include_content": True, "max_files": 8, "max_chars": 3000, "max_total_chars": 8000},
    "document": {"include_tree": False, "include_content": True, "max_files": 4, "max_chars": 3000, "max_total_chars": 8000},
}

PHASE_RESOURCE_POLICY: dict[str, dict[str, set[str]]] = {
    "phase0": {
        "include_classes": {"template", "library", "dataset", "model", "setup", "document"},
        "content_classes": {"template", "setup", "document"},
        "tree_classes": {"template", "setup"},
    },
    "phase1": {
        "include_classes": {"template", "library", "dataset", "model", "setup", "document"},
        "content_classes": {"template", "setup", "document"},
        "tree_classes": {"template", "setup"},
    },
    "phase2": {
        "include_classes": {"template", "library", "dataset", "model", "setup", "document"},
        "content_classes": {"template", "document"},
        "tree_classes": {"template"},
    },
    "phase3": {
        "include_classes": {"template", "library", "dataset", "model", "setup", "document"},
        "content_classes": {"document"},
        "tree_classes": set(),
    },
    "phase4": {
        "include_classes": {"template", "library", "dataset", "model", "setup", "document"},
        "content_classes": {"document"},
        "tree_classes": set(),
    },
}


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
class ResourceItem:
    """Classified resource entry for a specific file or directory."""

    name: str
    class_: ResourceClass
    source: ResourceSource
    resource: str
    path: str
    notes: str = ""
    include_tree: bool | None = None
    include_content: bool | None = None
    include_files: list[str] = field(default_factory=list)
    max_files: int | None = None
    max_chars: int | None = None
    max_total_chars: int | None = None

    def validate(self, local_map: dict[str, LocalResource], github_map: dict[str, GitHubResource], hf_map: dict[str, HuggingFaceResource]) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("ResourceItem.name is required")
        if self.class_ not in RESOURCE_CLASS_GUIDANCE:
            errors.append(f"ResourceItem '{self.name}': class must be one of {sorted(RESOURCE_CLASS_GUIDANCE.keys())}")
        if self.source not in ("local", "github", "huggingface"):
            errors.append(f"ResourceItem '{self.name}': source must be local|github|huggingface")
        if not self.resource:
            errors.append(f"ResourceItem '{self.name}': resource is required")
        if not self.path:
            errors.append(f"ResourceItem '{self.name}': path is required")
        if self.path.startswith("/"):
            errors.append(f"ResourceItem '{self.name}': path must be relative, not absolute: {self.path}")
        if ".." in Path(self.path).parts:
            errors.append(f"ResourceItem '{self.name}': path must not contain '..': {self.path}")
        if self.source == "local":
            res = local_map.get(self.resource)
            if not res:
                errors.append(f"ResourceItem '{self.name}': local resource '{self.resource}' not found")
            else:
                base = Path(res.host_path)
                target = base / self.path
                if not target.exists():
                    errors.append(f"ResourceItem '{self.name}': path does not exist under local resource '{self.resource}': {target}")
                for rel in self.include_files:
                    rel_path = Path(rel)
                    if rel_path.is_absolute():
                        errors.append(f"ResourceItem '{self.name}': include_files must be relative: {rel}")
                        continue
                    if ".." in rel_path.parts:
                        errors.append(f"ResourceItem '{self.name}': include_files must not contain '..': {rel}")
                        continue
                    inc_target = (base / self.path / rel_path).resolve()
                    if not inc_target.exists():
                        errors.append(f"ResourceItem '{self.name}': include file not found: {inc_target}")
        elif self.source == "github":
            if self.resource not in github_map:
                errors.append(f"ResourceItem '{self.name}': github resource '{self.resource}' not found")
        else:
            if self.resource not in hf_map:
                errors.append(f"ResourceItem '{self.name}': huggingface resource '{self.resource}' not found")
        return errors


@dataclass
class ResourceConfig:
    """Container for all resource definitions."""

    local: list[LocalResource] = field(default_factory=list)
    github: list[GitHubResource] = field(default_factory=list)
    huggingface: list[HuggingFaceResource] = field(default_factory=list)
    items: list[ResourceItem] = field(default_factory=list)

    def maps(self) -> tuple[dict[str, LocalResource], dict[str, GitHubResource], dict[str, HuggingFaceResource]]:
        return (
            {res.name: res for res in self.local},
            {res.name: res for res in self.github},
            {res.name: res for res in self.huggingface},
        )

    def validate(self) -> list[str]:
        """Validate all resources and return errors."""
        errors: list[str] = []
        for res in self.local:
            errors.extend(res.validate())
        for res in self.github:
            errors.extend(res.validate())
        for res in self.huggingface:
            errors.extend(res.validate())
        local_map, github_map, hf_map = self.maps()
        for item in self.items:
            errors.extend(item.validate(local_map, github_map, hf_map))
        return errors

    def has_resources(self) -> bool:
        """Check if any resources are defined."""
        return bool(self.local or self.github or self.huggingface or self.items)


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


def _parse_resource_item(data: dict[str, Any]) -> ResourceItem:
    return ResourceItem(
        name=data.get("name", ""),
        class_=str(data.get("class", "")).lower(),
        source=str(data.get("source", "")).lower(),
        resource=str(data.get("resource", "")),
        path=str(data.get("path", "")),
        notes=data.get("notes", ""),
        include_tree=data.get("include_tree"),
        include_content=data.get("include_content"),
        include_files=list(data.get("include_files", []) or []),
        max_files=data.get("max_files"),
        max_chars=data.get("max_chars"),
        max_total_chars=data.get("max_total_chars"),
    )


def resolve_resources_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    root = os.environ.get("AI_SCIENTIST_ROOT")
    if root:
        return (Path(root) / resolved).resolve()
    return resolved.resolve()


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
    path = resolve_resources_path(path)
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

    base_dir = path.parent

    local_resources: list[LocalResource] = []
    for entry in data.get("local", []):
        res = _parse_local_resource(entry)
        if res.host_path and not Path(res.host_path).is_absolute():
            res.host_path = str((base_dir / res.host_path).resolve())
        local_resources.append(res)

    config = ResourceConfig(
        local=local_resources,
        github=[_parse_github_resource(r) for r in data.get("github", [])],
        huggingface=[_parse_huggingface_resource(r) for r in data.get("huggingface", [])],
        items=[_parse_resource_item(r) for r in data.get("items", [])],
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


def _phase_policy(phase: str | None) -> dict[str, set[str]]:
    key = str(phase or "phase0").lower()
    return PHASE_RESOURCE_POLICY.get(key, PHASE_RESOURCE_POLICY["phase0"])


def _resource_maps(resources: ResourceConfig) -> tuple[dict[str, LocalResource], dict[str, GitHubResource], dict[str, HuggingFaceResource]]:
    return resources.maps()


def _resolve_item_paths(resources: ResourceConfig, item: ResourceItem) -> tuple[Path | None, Path | None]:
    local_map, github_map, hf_map = _resource_maps(resources)
    rel = Path(item.path or ".")
    if item.source == "local":
        res = local_map.get(item.resource)
        if not res:
            return None, None
        host_root = Path(res.host_path)
        container_root = Path(res.mount_path)
        return host_root / rel, container_root / rel
    if item.source == "github":
        res = github_map.get(item.resource)
        if not res:
            return None, None
        return None, Path(res.dest) / rel
    res = hf_map.get(item.resource)
    if not res:
        return None, None
    return None, Path(res.dest) / rel


def _is_text_file(path: Path, *, max_bytes: int = 200_000) -> bool:
    try:
        if path.stat().st_size > max_bytes:
            return False
        data = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _collect_tree(root: Path, *, max_depth: int = 3, max_entries: int = 200) -> list[str]:
    entries: list[str] = []
    if not root.exists():
        return entries
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        for name in sorted(dirnames):
            entries.append(str((rel_dir / name).as_posix()) + "/")
        for name in sorted(filenames):
            entries.append(str((rel_dir / name).as_posix()))
        if len(entries) >= max_entries:
            entries = entries[:max_entries]
            entries.append("... (tree truncated)")
            break
    return entries


def _select_files_for_content(base: Path, *, include_files: Sequence[str], max_files: int) -> list[Path]:
    if include_files:
        selected: list[Path] = []
        for rel in include_files:
            rel_path = Path(rel)
            if rel_path.is_absolute():
                continue
            candidate = base / rel_path
            if candidate.exists() and candidate.is_file():
                selected.append(candidate)
        return selected
    if max_files <= 0:
        return []
    include_ext = {
        ".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".ini", ".cfg",
        ".toml", ".cmake", ".mk", ".rst",
    }
    include_names = {"makefile", "cmakelists.txt", "requirements.txt", "setup.py"}
    selected: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in include_ext or name in include_names:
            if _is_text_file(path):
                selected.append(path)
        if len(selected) >= max_files:
            break
    return selected


def _summarize_file(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if max_chars <= 0:
        return text
    return summarize_text(text, max_lines=120, max_chars=max_chars)


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base).as_posix())
    except ValueError:
        return str(path)


def build_resources_context(
    resources: ResourceConfig | None,
    *,
    phase: str | None = None,
    include_host_paths: bool = False,
) -> dict[str, Any]:
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

    policy = _phase_policy(phase)
    items: list[dict[str, Any]] = []
    for item in resources.items:
        if item.class_ not in policy["include_classes"]:
            continue
        host_path, container_path = _resolve_item_paths(resources, item)
        defaults = RESOURCE_CLASS_DEFAULTS.get(item.class_, {})
        include_tree = item.include_tree if item.include_tree is not None else item.class_ in policy["tree_classes"]
        include_content = item.include_content if item.include_content is not None else item.class_ in policy["content_classes"]
        max_files = int(item.max_files) if item.max_files is not None else int(defaults.get("max_files", 0))
        max_chars = int(item.max_chars) if item.max_chars is not None else int(defaults.get("max_chars", 2000))
        max_total_chars = int(item.max_total_chars) if item.max_total_chars is not None else int(defaults.get("max_total_chars", 0))

        payload: dict[str, Any] = {
            "name": item.name,
            "class": item.class_,
            "source": item.source,
            "resource": item.resource,
            "path": item.path,
            "container_path": str(container_path) if container_path else "",
            "notes": item.notes,
        }
        if include_host_paths and host_path:
            payload["host_path"] = str(host_path)

        if host_path and host_path.exists():
            if include_tree and host_path.is_dir():
                payload["tree"] = _collect_tree(host_path)
            if include_content:
                base = host_path if host_path.is_dir() else host_path.parent
                if host_path.is_file():
                    files = [host_path]
                else:
                    files = _select_files_for_content(
                        base,
                        include_files=item.include_files,
                        max_files=max_files,
                    )
                total_chars = 0
                file_payloads: list[dict[str, str]] = []
                for fpath in files:
                    if not fpath.exists():
                        continue
                    excerpt = _summarize_file(fpath, max_chars=max_chars)
                    if not excerpt:
                        continue
                    if max_total_chars and total_chars + len(excerpt) > max_total_chars:
                        break
                    rel_path = _safe_relpath(fpath, base)
                    file_payloads.append({"path": rel_path, "content": excerpt})
                    total_chars += len(excerpt)
                if file_payloads:
                    payload["files"] = file_payloads

        items.append(payload)

    items_by_class: dict[str, list[dict[str, Any]]] = {}
    for entry in items:
        items_by_class.setdefault(entry.get("class", ""), []).append(entry)

    return {
        "has_resources": resources.has_resources(),
        "phase": phase or "phase0",
        "class_guidance": RESOURCE_CLASS_GUIDANCE,
        "notes": [
            "Local mounts are already bind-mounted (respect read-only settings).",
            "GitHub/HuggingFace resources must be fetched into their dest paths.",
        ],
        "local_mounts": local_mounts,
        "github_resources": github_resources,
        "huggingface_resources": hf_resources,
        "items": items,
        "items_by_class": items_by_class,
        "hf_home": "/workspace/.cache/huggingface",
        "install_paths": {
            "cmake": "/workspace/.local",
            "python": "/workspace/.pydeps",
        },
    }


def stage_resource_items(
    resources: ResourceConfig | None,
    dest_root: Path,
    *,
    classes: Sequence[str] = ("template", "setup", "document"),
) -> list[dict[str, str]]:
    if not resources:
        return []
    dest_root.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, str]] = []
    for item in resources.items:
        if item.source != "local":
            continue
        if item.class_ not in classes:
            continue
        host_path, _ = _resolve_item_paths(resources, item)
        if not host_path or not host_path.exists():
            continue
        dest = dest_root / item.class_ / item.name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if host_path.is_dir():
                shutil.copytree(host_path, dest)
            else:
                shutil.copy2(host_path, dest)
            staged.append({
                "name": item.name,
                "class": item.class_,
                "source": str(host_path),
                "dest": str(dest),
            })
        except OSError:
            continue
    return staged
