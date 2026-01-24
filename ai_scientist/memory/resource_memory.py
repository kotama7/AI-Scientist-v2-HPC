from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence
from ai_scientist.treesearch.utils.resource import (
    RESOURCE_CLASS_DEFAULTS,
    ResourceConfig,
    ResourceItem,
    load_resources,
    resolve_resources_path,
    _is_text_file,
    _resolve_item_paths,
)

logger = logging.getLogger(__name__)

RESOURCE_INDEX_KEY = "RESOURCE_INDEX"
RESOURCE_INDEX_JSON_KEY = "resource_index_json"
RESOURCE_DIGEST_KEY = "resource_digest"
RESOURCE_USED_KEY = "resource_used"

RESOURCE_ITEM_TAG = "RESOURCE_ITEM"
RESOURCE_PENDING_TAG = "RESOURCE_PENDING"
RESOURCE_USED_TAG = "RESOURCE_USED"

RESOURCE_INDEX_MAX_CHARS_DEFAULT = 2000
RESOURCE_ITEM_MAX_CHARS = 6000

STAGED_CLASSES = {"template", "setup", "document"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def _ensure_logger(log: Any | None) -> logging.Logger:
    if isinstance(log, logging.Logger):
        return log
    return logger


def _resolve_resource_file(path: str | Path, ai_scientist_root: str | Path | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if ai_scientist_root:
        return (Path(ai_scientist_root) / candidate).resolve()
    return resolve_resources_path(candidate)


def _map_container_to_host(
    container_path: Path | None,
    workspace_root: Path,
    *,
    workspace_mount: str = "/workspace",
) -> Path | None:
    if not container_path:
        return None
    raw = str(container_path)
    if raw.startswith(workspace_mount):
        rel = raw[len(workspace_mount) :].lstrip("/")
        return (workspace_root / rel).resolve()
    return None


def _normalize_config(
    cfg: ResourceConfig,
    resource_file: Path,
    resource_file_sha: str | None = None,
) -> dict[str, Any]:
    def sort_key(item: dict[str, Any]) -> tuple:
        return tuple(str(item.get(k, "")) for k in ("name", "class", "source", "resource", "path"))

    normalized = {
        "resource_file": str(resource_file),
        "resource_file_sha": resource_file_sha or "",
        "local": sorted(
            [
                {
                    "name": res.name,
                    "host_path": str(Path(res.host_path).resolve()) if res.host_path else "",
                    "mount_path": res.mount_path,
                    "read_only": res.read_only,
                }
                for res in cfg.local
            ],
            key=lambda item: item.get("name", ""),
        ),
        "github": sorted(
            [
                {
                    "name": res.name,
                    "repo": res.repo,
                    "ref": res.ref,
                    "dest": res.dest,
                    "as": res.as_,
                }
                for res in cfg.github
            ],
            key=lambda item: item.get("name", ""),
        ),
        "huggingface": sorted(
            [
                {
                    "name": res.name,
                    "type": res.type,
                    "repo_id": res.repo_id,
                    "revision": res.revision,
                    "dest": res.dest,
                }
                for res in cfg.huggingface
            ],
            key=lambda item: item.get("name", ""),
        ),
        "items": sorted(
            [
                {
                    "name": item.name,
                    "class": item.class_,
                    "source": item.source,
                    "resource": item.resource,
                    "path": item.path,
                    "notes": item.notes,
                    "include_tree": item.include_tree,
                    "include_content": item.include_content,
                    "include_files": list(item.include_files or []),
                    "max_files": item.max_files,
                    "max_chars": item.max_chars,
                    "max_total_chars": item.max_total_chars,
                }
                for item in cfg.items
            ],
            key=sort_key,
        ),
    }
    return normalized


def _stable_item_id(item: ResourceItem) -> str:
    return f"{item.class_}:{item.name}:{item.source}:{item.resource}:{item.path}"


def _summarize_head(text: str, *, max_lines: int, max_chars: int) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:max_lines])
    return _truncate(head, max_chars)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _collect_tree(
    root: Path,
    *,
    max_files: int,
    max_total_chars: int,
    max_depth: int = 4,
) -> list[str]:
    if not root.exists() or max_files <= 0:
        return []
    entries: list[str] = []
    total_chars = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(dirnames)
        filenames = sorted(filenames)
        for name in dirnames:
            entry = str((rel_dir / name).as_posix()) + "/"
            if max_files and len(entries) >= max_files:
                entries.append("... (tree truncated)")
                return entries
            if max_total_chars and total_chars + len(entry) > max_total_chars:
                entries.append("... (tree truncated)")
                return entries
            entries.append(entry)
            total_chars += len(entry)
        for name in filenames:
            entry = str((rel_dir / name).as_posix())
            if max_files and len(entries) >= max_files:
                entries.append("... (tree truncated)")
                return entries
            if max_total_chars and total_chars + len(entry) > max_total_chars:
                entries.append("... (tree truncated)")
                return entries
            entries.append(entry)
            total_chars += len(entry)
    return entries


def _select_content_files(
    base: Path,
    *,
    include_files: Sequence[str],
    max_files: int,
) -> list[Path]:
    if include_files:
        selected: list[Path] = []
        for rel in include_files:
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                continue
            candidate = base / rel_path
            if candidate.exists() and candidate.is_file():
                selected.append(candidate)
            if max_files and len(selected) >= max_files:
                break
        return selected
    if max_files <= 0:
        return []
    include_ext = {
        ".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".ini", ".cfg",
        ".toml", ".cmake", ".mk", ".rst",
    }
    include_names = {"makefile", "cmakelists.txt", "requirements.txt", "setup.py"}
    candidates: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in include_ext or name in include_names:
            if _is_text_file(path):
                candidates.append(path)
        if len(candidates) >= max_files:
            break
    return candidates


def _render_file_payload(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    summary = _summarize_head(text, max_lines=8, max_chars=min(400, max_chars // 2))
    excerpt_max = max_chars - len(summary) - 20
    excerpt = _truncate(text, max(120, excerpt_max))
    return f"Summary:\n{summary}\n\nExcerpt:\n{excerpt}"


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base).as_posix())
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ResourceFilePayload:
    path: str
    size: int
    mtime: int
    content: str


@dataclass(frozen=True)
class ResourceItemSnapshot:
    id: str
    name: str
    class_: str
    source: str
    resource: str
    original_path: str
    notes: str
    host_path: str | None
    container_path: str | None
    staged_path: str | None
    dest_path: str | None
    include_tree: bool
    include_content: bool
    include_files: list[str]
    max_files: int
    max_chars: int
    max_total_chars: int
    availability: str
    fetch_status: str
    metadata_only: bool
    resolved_paths: list[str]
    staging_info: dict[str, Any] | None
    tree_summary: str
    content_excerpt: str
    tree: list[str]
    files: list[ResourceFilePayload]
    digest: str


@dataclass(frozen=True)
class ResourceSnapshot:
    resource_file: str
    resource_file_sha: str
    normalized: dict[str, Any]
    items: list[ResourceItemSnapshot]
    resource_digest: str
    created_at: str
    phase_mode: str


def _item_digest_payload(item: ResourceItemSnapshot) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "class": item.class_,
        "source": item.source,
        "resource": item.resource,
        "original_path": item.original_path,
        "notes": item.notes,
        "staged_path": item.staged_path,
        "dest_path": item.dest_path,
        "container_path": item.container_path,
        "include_tree": item.include_tree,
        "include_content": item.include_content,
        "include_files": list(item.include_files or []),
        "max_files": item.max_files,
        "max_chars": item.max_chars,
        "max_total_chars": item.max_total_chars,
        "availability": item.availability,
        "fetch_status": item.fetch_status,
        "metadata_only": item.metadata_only,
        "resolved_paths": list(item.resolved_paths or []),
        "staging_info": item.staging_info or {},
        "tree_summary": item.tree_summary,
        "content_excerpt": item.content_excerpt,
        "tree": list(item.tree or []),
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "mtime": f.mtime,
                "content": f.content,
            }
            for f in item.files
        ],
    }


def _item_content_digest_payload(item: ResourceItemSnapshot) -> dict[str, Any]:
    """Compute digest payload excluding process-specific paths.

    This is used for duplicate detection in archival to avoid writing
    the same resource content multiple times from different worker processes.
    Excludes: staged_path, resolved_paths, staging_info (which contain process IDs).
    """
    return {
        "id": item.id,
        "name": item.name,
        "class": item.class_,
        "source": item.source,
        "resource": item.resource,
        "original_path": item.original_path,
        "notes": item.notes,
        "dest_path": item.dest_path,
        "container_path": item.container_path,
        "include_tree": item.include_tree,
        "include_content": item.include_content,
        "include_files": list(item.include_files or []),
        "max_files": item.max_files,
        "max_chars": item.max_chars,
        "max_total_chars": item.max_total_chars,
        "availability": item.availability,
        "fetch_status": item.fetch_status,
        "metadata_only": item.metadata_only,
        "tree_summary": item.tree_summary,
        "content_excerpt": item.content_excerpt,
        "tree": list(item.tree or []),
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "mtime": f.mtime,
                "content": f.content,
            }
            for f in item.files
        ],
    }


def _compute_content_digest(item: ResourceItemSnapshot) -> str:
    """Compute content-only digest for duplicate detection."""
    return _compute_digest(_item_content_digest_payload(item))


def _snapshot_digest_payload(snapshot: ResourceSnapshot) -> dict[str, Any]:
    return {
        "resource_file_sha": snapshot.resource_file_sha,
        "normalized": snapshot.normalized,
        "items": [_item_digest_payload(item) for item in snapshot.items],
    }


def _compute_digest(payload: dict[str, Any]) -> str:
    raw = _safe_json(payload)
    return sha256(raw.encode("utf-8")).hexdigest()


def _item_tags(item: ResourceItemSnapshot) -> list[str]:
    tags = [
        RESOURCE_ITEM_TAG,
        f"resource_id:{item.id}",
        f"resource:{item.class_}:{item.name}",
        f"resource_class:{item.class_}",
        f"resource_name:{item.name}",
        f"resource_source:{item.source}",
    ]
    if item.original_path:
        tags.append(f"resource_path:{item.original_path}")
    if item.container_path:
        tags.append(f"resource_container:{item.container_path}")
    if item.availability != "available":
        tags.append(RESOURCE_PENDING_TAG)
    return tags


def _index_items(snapshot: ResourceSnapshot) -> list[dict[str, Any]]:
    items = []
    for item in snapshot.items:
        items.append(
            {
                "id": item.id,
                "class": item.class_,
                "name": item.name,
                "source": item.source,
                "resource": item.resource,
                "original_path": item.original_path,
                "staged_path": item.staged_path,
                "container_path": item.container_path,
                "dest_path": item.dest_path,
                "availability": item.availability,
                "fetch_status": item.fetch_status,
                "metadata_only": item.metadata_only,
                "digest": item.digest,
                "tags": _item_tags(item),
            }
        )
    return items


def _format_resource_index(
    snapshot: ResourceSnapshot,
    *,
    previous_digest: str | None,
    max_chars: int,
) -> str:
    lines: list[str] = [
        "RESOURCE_INDEX",
        f"digest: {snapshot.resource_digest}",
        f"generated_at: {snapshot.created_at}",
        f"resource_file: {snapshot.resource_file}",
        f"resource_file_sha: {snapshot.resource_file_sha}",
    ]
    if previous_digest:
        lines.append(f"previous_digest: {previous_digest}")
    lines.append(f"normalized_config: {_safe_json(snapshot.normalized)}")
    lines.append("item_index:")
    for item in snapshot.items:
        staged = item.staged_path or "-"
        dest = item.dest_path or item.container_path or ""
        tags = ", ".join(_item_tags(item))
        lines.append(
            f"- {item.class_}/{item.name} ({item.source}) "
            f"avail={item.availability} fetch={item.fetch_status} digest={item.digest} "
            f"staged={staged} dest={dest} id={item.id} tags=[{tags}]"
        )
    lines.extend(
        [
            "how_to_use:",
            "- Search memory tag resource:<class>:<name> for resource content/excerpts.",
            "- Use resource_path:<path> or resource_source:<source> to narrow retrieval.",
        ]
    )
    return _truncate("\n".join(lines), max_chars)


def _render_item_body(item: ResourceItemSnapshot, *, snapshot_digest: str) -> str:
    lines: list[str] = [
        f"Resource: {item.name}",
        f"Class: {item.class_}",
        f"Source: {item.source}",
        f"Resource name: {item.resource}",
        f"Original path: {item.original_path}",
        f"Resource digest: {snapshot_digest}",
    ]
    if item.container_path:
        lines.append(f"Container path: {item.container_path}")
    if item.staged_path:
        lines.append(f"Staged path: {item.staged_path}")
    if item.dest_path:
        lines.append(f"Dest path: {item.dest_path}")
    lines.append(f"Availability: {item.availability}")
    lines.append(f"Fetch status: {item.fetch_status}")
    lines.append(f"Metadata only: {item.metadata_only}")
    if item.resolved_paths:
        lines.append(f"Resolved paths: {', '.join(item.resolved_paths)}")
    if item.staging_info:
        lines.append(f"Staging info: {_safe_json(item.staging_info)}")
    if item.availability != "available":
        lines.append("Status: pending fetch or missing content.")
    if item.tree_summary:
        lines.append("")
        lines.append("Tree summary:")
        lines.append(item.tree_summary)
    if item.content_excerpt:
        lines.append("")
        lines.append("Content excerpt:")
        lines.append(item.content_excerpt)
    return _truncate("\n".join(lines), RESOURCE_ITEM_MAX_CHARS)


def build_resource_snapshot(
    resource_file_path: str | Path,
    workspace_root: str | Path,
    ai_scientist_root: str | Path | None = None,
    phase_mode: str | None = None,
    log: Any | None = None,
) -> ResourceSnapshot:
    log = _ensure_logger(log)
    workspace_root = Path(workspace_root).resolve()
    resolved_path = _resolve_resource_file(resource_file_path, ai_scientist_root)
    log.debug("Building resource snapshot for %s", resolved_path)
    resource_file_text = _read_text(resolved_path)
    resource_file_sha = sha256(resource_file_text.encode("utf-8")).hexdigest()
    cfg = load_resources(resolved_path)
    normalized = _normalize_config(cfg, resolved_path, resource_file_sha=resource_file_sha)
    items: list[ResourceItemSnapshot] = []
    _, github_map, hf_map = cfg.maps()

    for item in sorted(cfg.items, key=lambda i: (i.class_, i.name, i.source, i.resource, i.path)):
        defaults = RESOURCE_CLASS_DEFAULTS.get(item.class_, {})
        include_tree = item.include_tree if item.include_tree is not None else bool(defaults.get("include_tree", False))
        include_content = item.include_content if item.include_content is not None else bool(defaults.get("include_content", False))
        max_files = int(item.max_files) if item.max_files is not None else int(defaults.get("max_files", 0))
        max_chars = int(item.max_chars) if item.max_chars is not None else int(defaults.get("max_chars", 2000))
        max_total_chars = int(item.max_total_chars) if item.max_total_chars is not None else int(defaults.get("max_total_chars", 0))

        host_path, container_path = _resolve_item_paths(cfg, item)
        host_path_resolved = host_path.resolve() if host_path else None

        dest_root = None
        if item.source == "github":
            res = github_map.get(item.resource)
            if res:
                dest_root = Path(res.dest)
        elif item.source == "huggingface":
            res = hf_map.get(item.resource)
            if res:
                dest_root = Path(res.dest)
        dest_path = _map_container_to_host(dest_root, workspace_root) if dest_root else None

        staged_path = None
        if item.source == "local" and item.class_ in STAGED_CLASSES:
            staged_path = str((workspace_root / "resources" / item.class_ / item.name).resolve())

        availability = "available"
        if item.source == "local":
            if not host_path_resolved or not host_path_resolved.exists():
                availability = "missing"
        else:
            if dest_path is None or not dest_path.exists():
                availability = "pending_fetch"
            else:
                host_path_resolved = _map_container_to_host(container_path, workspace_root)
                if host_path_resolved is None or not host_path_resolved.exists():
                    availability = "missing"

        tree: list[str] = []
        files: list[ResourceFilePayload] = []
        if availability == "available" and host_path_resolved and host_path_resolved.exists():
            if include_tree and host_path_resolved.is_dir():
                tree = _collect_tree(
                    host_path_resolved,
                    max_files=max_files,
                    max_total_chars=max_total_chars,
                )
            if include_content:
                base = host_path_resolved if host_path_resolved.is_dir() else host_path_resolved.parent
                if host_path_resolved.is_file():
                    selected_files = [host_path_resolved]
                else:
                    selected_files = _select_content_files(
                        base,
                        include_files=item.include_files or [],
                        max_files=max_files,
                    )
                total_chars = 0
                for fpath in selected_files:
                    if not fpath.exists() or not _is_text_file(fpath):
                        continue
                    stat = fpath.stat()
                    text = _read_text(fpath)
                    payload = _render_file_payload(text, max_chars=max_chars)
                    if not payload:
                        continue
                    if max_total_chars and total_chars + len(payload) > max_total_chars:
                        break
                    rel_path = _safe_relpath(fpath, base)
                    files.append(
                        ResourceFilePayload(
                            path=rel_path,
                            size=int(stat.st_size),
                            mtime=int(stat.st_mtime),
                            content=payload,
                        )
                    )
                    total_chars += len(payload)

        fetch_status = "available"
        if availability == "pending_fetch":
            fetch_status = "pending"
        elif availability != "available":
            fetch_status = "failed"

        metadata_only = not include_tree and not include_content and not (item.include_files or [])
        resolved_paths = [
            path
            for path in (
                str(container_path) if container_path else None,
                staged_path,
                str(dest_path) if dest_path else None,
            )
            if path
        ]
        staging_info = None
        if staged_path:
            staging_info = {
                "class": item.class_,
                "name": item.name,
                "path": staged_path,
            }
        tree_summary = ""
        if tree:
            tree_summary = _truncate("\n".join(tree), max_total_chars or max_chars)
        content_excerpt = ""
        if files:
            excerpt_lines: list[str] = []
            for f in files:
                excerpt_lines.append(f"File: {f.path} (size={f.size}, mtime={f.mtime})")
                if f.content:
                    excerpt_lines.append(f.content)
            excerpt_limit = max_total_chars or max(max_chars * 2, 2000)
            content_excerpt = _truncate("\n".join(excerpt_lines), excerpt_limit)

        snapshot_item = ResourceItemSnapshot(
            id=_stable_item_id(item),
            name=item.name,
            class_=item.class_,
            source=item.source,
            resource=item.resource,
            original_path=item.path,
            notes=item.notes,
            host_path=str(host_path_resolved) if host_path_resolved else None,
            container_path=str(container_path) if container_path else None,
            staged_path=staged_path,
            dest_path=str(dest_path) if dest_path else None,
            include_tree=include_tree,
            include_content=include_content,
            include_files=list(item.include_files or []),
            max_files=max_files,
            max_chars=max_chars,
            max_total_chars=max_total_chars,
            availability=availability,
            fetch_status=fetch_status,
            metadata_only=metadata_only,
            resolved_paths=resolved_paths,
            staging_info=staging_info,
            tree_summary=tree_summary,
            content_excerpt=content_excerpt,
            tree=tree,
            files=files,
            digest="",
        )
        item_digest = _compute_digest(_item_digest_payload(snapshot_item))
        items.append(
            ResourceItemSnapshot(
                **{**snapshot_item.__dict__, "digest": item_digest},
            )
        )

    snapshot = ResourceSnapshot(
        resource_file=str(resolved_path),
        resource_file_sha=resource_file_sha,
        normalized=normalized,
        items=items,
        resource_digest="",
        created_at=_now_iso(),
        phase_mode=str(phase_mode or "unknown"),
    )
    digest = _compute_digest(_snapshot_digest_payload(snapshot))
    return ResourceSnapshot(
        **{**snapshot.__dict__, "resource_digest": digest},
    )


def _ltm_branch_id(ltm: Any, override: str | None = None) -> str | None:
    if override:
        return override
    if hasattr(ltm, "config"):
        return getattr(ltm.config, "root_branch_id", None)
    return None


def _write_snapshot_files(snapshot: ResourceSnapshot, ltm: Any) -> None:
    memory_dir = Path(getattr(ltm, "db_path", Path("."))).parent
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "resource_digest": snapshot.resource_digest,
            "resource_file": snapshot.resource_file,
            "resource_file_sha": snapshot.resource_file_sha,
            "normalized": snapshot.normalized,
            "items": _index_items(snapshot),
            "created_at": snapshot.created_at,
            "phase_mode": snapshot.phase_mode,
        }
        (memory_dir / "resource_snapshot.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write resource snapshot file: %s", exc)


def track_resource_usage(name_or_id: str, context: dict[str, Any] | None) -> None:
    if not context:
        return
    ltm = context.get("ltm")
    if ltm is None:
        return
    root_id = _ltm_branch_id(ltm, None)
    branch_id = root_id or context.get("branch_id")
    if not branch_id:
        return
    note = str(context.get("note") or "").strip()
    try:
        raw = ltm.get_core(branch_id, RESOURCE_USED_KEY)
        used = json.loads(raw) if raw else []
        if not isinstance(used, list):
            used = []
    except Exception:
        used = []
    if name_or_id in used:
        return
    used.append(name_or_id)
    used = sorted(set(used))
    if hasattr(ltm, "mem_core_set") and getattr(ltm, "root_branch_id", None):
        ltm.mem_core_set(
            RESOURCE_USED_KEY,
            json.dumps(used, sort_keys=True, ensure_ascii=True),
            importance=4,
        )
    else:
        ltm.set_core(branch_id, RESOURCE_USED_KEY, json.dumps(used, sort_keys=True, ensure_ascii=True))
    text = f"{name_or_id}"
    if note:
        text = f"{text} | {note}"
    try:
        if hasattr(ltm, "mem_recall_append"):
            ltm.mem_recall_append(
                {
                    "ts": time.time(),
                    "run_id": getattr(ltm, "run_id", ""),
                    "node_id": branch_id,
                    "phase": "resource_usage",
                    "kind": "resource_used",
                    "summary": text,
                    "refs": [],
                }
            )
        else:
            ltm.write_event(
                branch_id,
                "resource_used",
                text,
                tags=[RESOURCE_USED_TAG, f"resource_id:{name_or_id}"],
            )
    except Exception:
        return
