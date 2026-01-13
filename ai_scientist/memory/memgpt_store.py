from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence

from .resource_memory import (
    RESOURCE_DIGEST_KEY,
    RESOURCE_INDEX_JSON_KEY,
    RESOURCE_INDEX_KEY,
    RESOURCE_ITEM_TAG,
    RESOURCE_USED_KEY,
    RESOURCE_USED_TAG,
    build_resource_snapshot,
    track_resource_usage,
    update_resource_snapshot_if_changed,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _now_ts() -> float:
    return time.time()


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = [tags]
    elif isinstance(tags, (list, tuple, set)):
        raw = list(tags)
    else:
        raw = [str(tags)]
    cleaned = []
    for tag in raw:
        tag_str = str(tag).strip()
        if tag_str:
            cleaned.append(tag_str)
    return cleaned


_SECRET_PATTERNS = [
    re.compile(r"(?i)(api_key|secret|token|password)\s*[:=]\s*([^\s]+)"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
]


def _redact(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}=***" if m.groups() else "***", redacted)
    return redacted


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=True)
        return value
    except Exception:
        return str(value)


def _core_lines(entries: dict[str, str]) -> list[str]:
    return [f"{key}: {value}" for key, value in sorted(entries.items(), key=lambda kv: kv[0])]


def _core_size(entries: dict[str, str]) -> int:
    return len("\n".join(_core_lines(entries)))


def _coerce_importance(value: Any) -> int:
    try:
        importance = int(value)
    except Exception:
        importance = 3
    return max(1, min(5, importance))


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("\n", 1)
    if len(parts) < 2:
        return text
    rest = parts[1]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return text
    return rest[end_idx + len("\n---") :].lstrip()


def _extract_resource_ids(tags_json: str | None) -> list[str]:
    try:
        tags = json.loads(tags_json) if tags_json else []
    except json.JSONDecodeError:
        tags = []
    ids: list[str] = []
    for tag in tags:
        tag_str = str(tag)
        if tag_str.startswith("resource_id:"):
            ids.append(tag_str.split("resource_id:", 1)[1])
    if not ids:
        for tag in tags:
            tag_str = str(tag)
            if tag_str.startswith("resource:"):
                ids.append(tag_str)
    return ids


def _parse_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("### ") and current is None:
            current = line[4:].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _summarize_idea(text: str, max_chars: int = 800) -> str:
    sections = _parse_markdown_sections(text)
    purpose = sections.get("Abstract") or sections.get("Task goal") or ""
    hypothesis = sections.get("Short Hypothesis") or sections.get("Hypothesis") or ""
    method = sections.get("Experiments") or sections.get("Code") or ""
    evaluation = sections.get("Task evaluation") or sections.get("Evaluation") or ""
    risks = sections.get("Risk Factors And Limitations") or sections.get(
        "Risk Factors and Limitations"
    ) or ""

    def pick(src: str, limit: int = 240) -> str:
        return _truncate(" ".join(src.split()), limit)

    bullets = [
        f"- Purpose: {pick(purpose)}" if purpose else "- Purpose: (not provided)",
        f"- Hypothesis: {pick(hypothesis)}" if hypothesis else "- Hypothesis: (not provided)",
        f"- Method/Variables: {pick(method)}" if method else "- Method/Variables: (not provided)",
        f"- Evaluation: {pick(evaluation)}" if evaluation else "- Evaluation: (not provided)",
        f"- Known failures/mitigations: {pick(risks)}" if risks else "- Known failures/mitigations: (not provided)",
    ]
    summary = "\n".join(bullets)
    return _truncate(summary, max_chars)


def _summarize_phase0(payload: Any, command_str: str | None, max_chars: int = 800) -> str:
    items: list[str] = []
    if isinstance(payload, dict):
        for key in (
            "threads",
            "thread",
            "pinning",
            "numa",
            "affinity",
            "task_placement",
            "omp_num_threads",
            "omp_proc_bind",
            "omp_places",
            "mkl_num_threads",
            "openblas_num_threads",
            "repeat",
            "warmup",
            "size",
            "seed",
            "compiler_selected",
        ):
            if key in payload:
                items.append(f"{key}={payload[key]}")
        build_plan = payload.get("plan") or payload.get("build_plan")
        if isinstance(build_plan, dict):
            for key in ("compiler_selected", "cflags", "ldflags", "workdir", "output"):
                if key in build_plan:
                    items.append(f"{key}={build_plan[key]}")
    if command_str:
        items.append(f"command={command_str}")
    if not items:
        items.append("No structured Phase 0 info captured.")
    return _truncate(" | ".join(items), max_chars)


class MemoryManager:
    def __init__(self, db_path: str | Path, config: Any | None = None):
        self.config = config or {}
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = _cfg_get(self.config, "run_id", None)
        self.run_id = str(run_id).strip() if run_id else None
        workspace_root = _cfg_get(self.config, "workspace_root", None)
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        ai_scientist_root = _cfg_get(self.config, "ai_scientist_root", None)
        self.ai_scientist_root = (
            Path(ai_scientist_root).resolve() if ai_scientist_root else None
        )
        self.root_branch_id = _cfg_get(self.config, "root_branch_id", None)
        self.core_max_chars = int(_cfg_get(self.config, "core_max_chars", 2000))
        self.recall_max_events = int(_cfg_get(self.config, "recall_max_events", 20))
        self.retrieval_k = int(_cfg_get(self.config, "retrieval_k", 8))
        self.use_fts_setting = str(_cfg_get(self.config, "use_fts", "auto")).lower()
        self.always_inject_phase0_summary = bool(
            _cfg_get(self.config, "always_inject_phase0_summary", True)
        )
        self.always_inject_idea_summary = bool(
            _cfg_get(self.config, "always_inject_idea_summary", True)
        )
        self.redact_secrets = bool(_cfg_get(self.config, "redact_secrets", True))
        self.memory_log_enabled = bool(_cfg_get(self.config, "memory_log_enabled", True))
        self.memory_log_max_chars = int(_cfg_get(self.config, "memory_log_max_chars", 400))
        self.memory_log_dir: Path | None = None
        self.memory_log_path: Path | None = None
        log_dir = _cfg_get(self.config, "memory_log_dir", None)
        if not log_dir:
            base_log_dir = _cfg_get(self.config, "log_dir", None)
            if base_log_dir:
                log_dir = str(Path(base_log_dir) / "memory")
        if log_dir:
            try:
                self.memory_log_dir = Path(log_dir)
                self.memory_log_dir.mkdir(parents=True, exist_ok=True)
                self.memory_log_path = self.memory_log_dir / "memory_calls.jsonl"
            except Exception:
                self.memory_log_dir = None
                self.memory_log_path = None
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=30, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._fts_enabled = self._init_fts()

    def _sanitize_detail_value(self, value: Any) -> Any:
        if isinstance(value, str):
            redacted = _redact(value, self.redact_secrets)
            return _truncate(redacted, self.memory_log_max_chars)
        if isinstance(value, dict):
            return {k: self._sanitize_detail_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_detail_value(v) for v in value]
        return _safe_json_value(value)

    def _log_memory_event(
        self,
        op: str,
        memory_type: str,
        *,
        branch_id: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.memory_log_enabled or not self.memory_log_path:
            return
        payload: dict[str, Any] = {
            "ts": _now_ts(),
            "op": op,
            "memory_type": memory_type,
            "branch_id": branch_id,
            "node_id": node_id,
            "phase": phase,
            "run_id": self.run_id,
        }
        if details:
            payload["details"] = self._sanitize_detail_value(details)
        try:
            self.memory_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.memory_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning("Failed to write memory log event: %s", exc)

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                parent_id TEXT NULL,
                node_uid TEXT NULL,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS core_kv (
                branch_id TEXT,
                key TEXT,
                value TEXT,
                updated_at REAL,
                PRIMARY KEY (branch_id, key)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS core_meta (
                branch_id TEXT,
                key TEXT,
                importance INTEGER,
                ttl TEXT,
                updated_at REAL,
                PRIMARY KEY (branch_id, key)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_id TEXT,
                kind TEXT,
                text TEXT,
                tags TEXT,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS archival (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_id TEXT,
                text TEXT,
                tags TEXT,
                created_at REAL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_archival_branch ON archival(branch_id)"
        )
        self._conn.commit()

    def _init_fts(self) -> bool:
        if self.use_fts_setting == "off":
            return False
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS archival_fts USING fts5(text, tags, branch_id)"
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError as exc:
            if self.use_fts_setting == "on":
                raise RuntimeError("FTS5 requested but unavailable") from exc
            logger.warning("FTS5 unavailable, falling back to keyword search.")
            return False

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def create_branch(
        self,
        parent_branch_id: str | None,
        node_uid: str | None,
        branch_id: str | None = None,
    ) -> str:
        branch_id = branch_id or uuid.uuid4().hex
        existing = self._conn.execute(
            "SELECT id FROM branches WHERE id=?", (branch_id,)
        ).fetchone()
        if existing:
            return branch_id
        self._conn.execute(
            "INSERT INTO branches (id, parent_id, node_uid, created_at) VALUES (?, ?, ?, ?)",
            (branch_id, parent_branch_id, node_uid, _now_ts()),
        )
        self._conn.commit()
        if parent_branch_id is None and not self.root_branch_id:
            self.set_root_branch_id(branch_id)
        return branch_id

    def update_branch_node_uid(self, branch_id: str, node_uid: str) -> None:
        self._conn.execute(
            "UPDATE branches SET node_uid=? WHERE id=?",
            (node_uid, branch_id),
        )
        self._conn.commit()

    def set_root_branch_id(self, branch_id: str) -> None:
        self.root_branch_id = branch_id
        try:
            if hasattr(self.config, "root_branch_id"):
                setattr(self.config, "root_branch_id", branch_id)
        except Exception:
            pass

    def _default_branch_id(self) -> str | None:
        root = self.root_branch_id
        if root:
            return root
        return _cfg_get(self.config, "root_branch_id", None)

    def _branch_exists(self, branch_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM branches WHERE id=?", (branch_id,)
        ).fetchone()
        return bool(row)

    def _resolve_branch_id(self, node_id: str | None) -> str | None:
        if not node_id:
            return None
        if self._branch_exists(node_id):
            return node_id
        row = self._conn.execute(
            "SELECT id FROM branches WHERE node_uid=?", (node_id,)
        ).fetchone()
        return row["id"] if row else None

    def _branch_chain(self, branch_id: str) -> list[str]:
        chain = []
        seen = set()
        current = branch_id
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            row = self._conn.execute(
                "SELECT parent_id FROM branches WHERE id=?", (current,)
            ).fetchone()
            current = row["parent_id"] if row else None
        return chain

    def _set_core_value(self, branch_id: str, key: str, value: str) -> None:
        text = _redact(str(value), self.redact_secrets)
        self._conn.execute(
            "INSERT OR REPLACE INTO core_kv (branch_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
            (branch_id, key, text, _now_ts()),
        )

    def _set_core_meta(self, branch_id: str, key: str, *, importance: int, ttl: str | None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO core_meta (branch_id, key, importance, ttl, updated_at) VALUES (?, ?, ?, ?, ?)",
            (branch_id, key, int(importance), ttl, _now_ts()),
        )

    def _delete_core_key(
        self,
        branch_id: str,
        key: str,
        *,
        reason: str | None = None,
        op_name: str | None = None,
        phase: str | None = None,
        node_id: str | None = None,
        log_event: bool = True,
    ) -> None:
        self._conn.execute(
            "DELETE FROM core_kv WHERE branch_id=? AND key=?",
            (branch_id, key),
        )
        self._conn.execute(
            "DELETE FROM core_meta WHERE branch_id=? AND key=?",
            (branch_id, key),
        )
        if log_event:
            self._log_memory_event(
                op_name or "core_delete",
                "core",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={"key": key, "reason": reason or ""},
            )

    def _enforce_core_budget(self, branch_id: str) -> None:
        if self.core_max_chars <= 0:
            return
        rows = self._conn.execute(
            "SELECT key, value FROM core_kv WHERE branch_id=?",
            (branch_id,),
        ).fetchall()
        entries = {row["key"]: row["value"] for row in rows}
        if not entries:
            return
        current_size = _core_size(entries)
        if current_size <= self.core_max_chars:
            return

        meta_rows = self._conn.execute(
            "SELECT key, importance FROM core_meta WHERE branch_id=?",
            (branch_id,),
        ).fetchall()
        importance_map = {row["key"]: _coerce_importance(row["importance"]) for row in meta_rows}
        protected = {
            RESOURCE_INDEX_KEY,
            RESOURCE_DIGEST_KEY,
            "idea_md_summary",
            "phase0_summary",
        }
        removable_keys = [
            (importance_map.get(key, 3), key)
            for key in entries
            if key not in protected
        ]
        removable_keys.sort(key=lambda pair: (pair[0], pair[1]))
        for _, key in removable_keys:
            value = entries.pop(key, "")
            if value:
                try:
                    self._insert_archival(
                        branch_id,
                        f"Core evicted: {key}\n{value}",
                        tags=[f"CORE_EVICT", f"core_key:{key}"],
                    )
                except Exception:
                    pass
            self._delete_core_key(
                branch_id,
                key,
                reason="evict",
                op_name="core_evict",
            )
            current_size = _core_size(entries)
            if current_size <= self.core_max_chars:
                self._conn.commit()
                return

        # Still too large; summarize remaining core into a digest.
        snapshot_text = "\n".join(_core_lines(entries))
        record_id = None
        if snapshot_text:
            try:
                record_id = self._insert_archival(
                    branch_id,
                    snapshot_text,
                    tags=["CORE_SNAPSHOT", f"branch_id:{branch_id}"],
                )
            except Exception:
                record_id = None
        resource_index = entries.get(RESOURCE_INDEX_KEY, "")
        for key in list(entries.keys()):
            if key == RESOURCE_INDEX_KEY:
                continue
            self._delete_core_key(
                branch_id,
                key,
                reason="digest",
                op_name="core_digest_compact",
            )
        available = self.core_max_chars - len(resource_index)
        if resource_index:
            available -= len("CoreDigest: ")
        digest_body = _truncate(snapshot_text, max(0, available))
        if digest_body:
            digest_value = digest_body
            if record_id:
                digest_value = f"{digest_body}\n(ref: {record_id})"
            self._set_core_value(branch_id, "CoreDigest", digest_value)
            self._set_core_meta(branch_id, "CoreDigest", importance=5, ttl=None)
        self._conn.commit()

    def set_core(
        self,
        branch_id: str,
        key: str,
        value: str,
        *,
        ttl: str | None = None,
        importance: int | None = None,
        op_name: str | None = None,
        phase: str | None = None,
        node_id: str | None = None,
        log_event: bool = True,
    ) -> None:
        if value is None:
            return
        self._set_core_value(branch_id, key, value)
        importance_val = _coerce_importance(importance if importance is not None else 3)
        self._set_core_meta(branch_id, key, importance=importance_val, ttl=ttl)
        if log_event:
            value_str = str(value)
            self._log_memory_event(
                op_name or "set_core",
                "core",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "key": key,
                    "value_preview": _truncate(
                        _redact(value_str, self.redact_secrets),
                        self.memory_log_max_chars,
                    ),
                    "value_chars": len(value_str),
                    "importance": importance_val,
                    "ttl": ttl,
                },
            )
        self._enforce_core_budget(branch_id)
        self._conn.commit()

    def get_core(
        self,
        branch_id: str,
        key: str,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        phase: str | None = None,
        node_id: str | None = None,
    ) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM core_kv WHERE branch_id=? AND key=?",
            (branch_id, key),
        ).fetchone()
        value = row["value"] if row else None
        if log_event:
            self._log_memory_event(
                op_name or "get_core",
                "core",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={"key": key, "found": value is not None},
            )
        return value

    def mem_core_get(self, keys: list[str] | None) -> dict[str, str]:
        branch_id = self._default_branch_id()
        if not branch_id:
            return {}
        if not keys:
            rows = self._conn.execute(
                "SELECT key, value FROM core_kv WHERE branch_id=?",
                (branch_id,),
            ).fetchall()
            result = {row["key"]: row["value"] for row in rows}
            self._log_memory_event(
                "mem_core_get",
                "core",
                branch_id=branch_id,
                details={"keys": None, "count": len(result)},
            )
            return result
        result = {key: self.get_core(branch_id, key, log_event=False) or "" for key in keys}
        self._log_memory_event(
            "mem_core_get",
            "core",
            branch_id=branch_id,
            details={"keys": keys, "count": len(result)},
        )
        return result

    def mem_core_set(
        self,
        key: str,
        value: str,
        *,
        ttl: str | None = None,
        importance: int = 3,
    ) -> None:
        branch_id = self._default_branch_id()
        if not branch_id:
            raise ValueError("mem_core_set requires a root branch id")
        self.set_core(
            branch_id,
            key,
            value,
            ttl=ttl,
            importance=importance,
            op_name="mem_core_set",
        )

    def mem_core_del(self, key: str) -> None:
        branch_id = self._default_branch_id()
        if not branch_id:
            return
        self._delete_core_key(
            branch_id,
            key,
            reason="explicit",
            op_name="mem_core_del",
        )
        self._conn.commit()

    def write_event(
        self,
        branch_id: str,
        kind: str,
        text: str,
        tags: Any = None,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
    ) -> None:
        payload = _redact(str(text), self.redact_secrets)
        tag_list = _normalize_tags(tags)
        self._conn.execute(
            "INSERT INTO events (branch_id, kind, text, tags, created_at) VALUES (?, ?, ?, ?, ?)",
            (branch_id, kind, payload, json.dumps(tag_list), _now_ts()),
        )
        self._conn.commit()
        if log_event:
            if not node_id or not phase:
                for tag in tag_list:
                    if not node_id and str(tag).startswith("node_id:"):
                        node_id = str(tag).split("node_id:", 1)[1]
                    if not phase and str(tag).startswith("phase:"):
                        phase = str(tag).split("phase:", 1)[1]
            self._log_memory_event(
                op_name or "write_event",
                "recall",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "kind": kind,
                    "summary_preview": _truncate(payload, self.memory_log_max_chars),
                    "tags": tag_list,
                },
            )

    def _insert_archival(
        self,
        branch_id: str,
        text: str,
        tags: Any = None,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
    ) -> int | None:
        payload = _redact(str(text), self.redact_secrets)
        tag_list = _normalize_tags(tags)
        cur = self._conn.execute(
            "INSERT INTO archival (branch_id, text, tags, created_at) VALUES (?, ?, ?, ?)",
            (branch_id, payload, json.dumps(tag_list), _now_ts()),
        )
        row_id = cur.lastrowid
        if self._fts_enabled and row_id:
            self._conn.execute(
                "INSERT INTO archival_fts (rowid, text, tags, branch_id) VALUES (?, ?, ?, ?)",
                (row_id, payload, json.dumps(tag_list), branch_id),
            )
        self._conn.commit()
        if log_event:
            self._log_memory_event(
                op_name or "write_archival",
                "archival",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "record_id": row_id,
                    "tags": tag_list,
                    "text_preview": _truncate(payload, self.memory_log_max_chars),
                    "text_chars": len(str(text)),
                },
            )
        return row_id

    def write_archival(self, branch_id: str, text: str, tags: Any = None) -> None:
        self._insert_archival(branch_id, text, tags=tags, op_name="write_archival")

    def _fetch_events(self, branch_ids: Sequence[str], limit: int) -> list[sqlite3.Row]:
        if not branch_ids:
            return []
        placeholders = ",".join(["?"] * len(branch_ids))
        query = (
            f"SELECT kind, text, tags, created_at FROM events "
            f"WHERE branch_id IN ({placeholders}) "
            "ORDER BY created_at DESC LIMIT ?"
        )
        rows = self._conn.execute(query, [*branch_ids, limit]).fetchall()
        return rows

    def retrieve_archival(
        self,
        branch_id: str,
        query: str | None,
        k: int | None = None,
        include_ancestors: bool = True,
        tags_filter: Any = None,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
    ) -> list[dict[str, Any]]:
        if not branch_id:
            return []
        k = k or self.retrieval_k
        branch_ids = self._branch_chain(branch_id) if include_ancestors else [branch_id]
        if not branch_ids:
            return []
        tag_list = _normalize_tags(tags_filter)
        placeholders = ",".join(["?"] * len(branch_ids))
        cleaned_query = (query or "").strip()
        rows: list[sqlite3.Row] = []
        if self._fts_enabled and cleaned_query:
            sql = (
                "SELECT a.id, a.branch_id, a.text, a.tags, a.created_at "
                "FROM archival_fts f JOIN archival a ON a.id = f.rowid "
                f"WHERE a.branch_id IN ({placeholders}) AND f.text MATCH ? "
                "ORDER BY a.created_at DESC LIMIT ?"
            )
            rows = self._conn.execute(sql, [*branch_ids, cleaned_query, k]).fetchall()
        else:
            sql = (
                "SELECT id, branch_id, text, tags, created_at FROM archival "
                f"WHERE branch_id IN ({placeholders}) "
                "ORDER BY created_at DESC"
            )
            rows = self._conn.execute(sql, branch_ids).fetchall()

        def tag_ok(tags_json: str | None) -> bool:
            if not tag_list:
                return True
            try:
                tags = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                tags = []
            return all(tag in tags for tag in tag_list)

        if not cleaned_query:
            filtered = [row for row in rows if tag_ok(row["tags"])]
            results = [dict(row) for row in filtered[:k]]
            if log_event:
                self._log_memory_event(
                    op_name or "retrieve_archival",
                    "archival",
                    branch_id=branch_id,
                    node_id=node_id,
                    phase=phase,
                    details={
                        "query": "",
                        "k": k,
                        "results": len(results),
                        "include_ancestors": include_ancestors,
                        "tags_filter": tag_list,
                    },
                )
            return results

        tokens = [t.lower() for t in re.split(r"\W+", cleaned_query) if t]
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if not tag_ok(row["tags"]):
                continue
            text = row["text"] or ""
            text_l = text.lower()
            score = 0.0
            for token in tokens:
                if token and token in text_l:
                    score += 1.0
            score += (row["created_at"] or 0) * 1e-12
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [dict(row) for _, row in scored[:k]]
        if log_event:
            self._log_memory_event(
                op_name or "retrieve_archival",
                "archival",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "query": cleaned_query,
                    "k": k,
                    "results": len(results),
                    "include_ancestors": include_ancestors,
                    "tags_filter": tag_list,
                },
            )
        return results

    def mem_recall_append(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        payload = dict(event)
        payload.setdefault("ts", _now_ts())
        if not payload.get("run_id") and self.run_id:
            payload["run_id"] = self.run_id
        node_id = payload.get("node_id")
        branch_id = self._resolve_branch_id(node_id) or self._default_branch_id()
        if not branch_id:
            return
        kind = str(payload.get("kind") or "event")
        summary = str(payload.get("summary") or "")
        refs = payload.get("refs") or []
        phase = str(payload.get("phase") or "")
        tags = [
            "RECALL_EVENT",
            f"run_id:{payload.get('run_id')}" if payload.get("run_id") else None,
            f"node_id:{node_id}" if node_id else None,
            f"phase:{phase}" if phase else None,
            f"kind:{kind}" if kind else None,
        ]
        for ref in refs if isinstance(refs, Iterable) and not isinstance(refs, str) else []:
            tags.append(f"ref:{ref}")
        tag_list = [tag for tag in tags if tag]
        self.write_event(
            branch_id,
            kind,
            summary,
            tags=tag_list,
            log_event=False,
        )
        self._log_memory_event(
            "mem_recall_append",
            "recall",
            branch_id=branch_id,
            node_id=node_id,
            phase=phase,
            details={
                "kind": kind,
                "summary_preview": _truncate(
                    _redact(summary, self.redact_secrets),
                    self.memory_log_max_chars,
                ),
                "tags": tag_list,
            },
        )

    def mem_recall_search(self, query: str, *, k: int = 20) -> list[dict]:
        branch_id = self._default_branch_id()
        if not branch_id:
            return []
        branch_ids = self._branch_chain(branch_id)
        rows = self._fetch_events(branch_ids, limit=max(k * 5, k))
        tokens = [t.lower() for t in re.split(r"\W+", query or "") if t]
        results: list[dict] = []
        for row in rows:
            text = (row["text"] or "")
            text_l = text.lower()
            if tokens and not any(token in text_l for token in tokens):
                continue
            tags_json = row["tags"]
            try:
                tag_list = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                tag_list = []
            meta = {"tags": tag_list}
            for tag in tag_list:
                if tag.startswith("run_id:"):
                    meta["run_id"] = tag.split("run_id:", 1)[1]
                if tag.startswith("node_id:"):
                    meta["node_id"] = tag.split("node_id:", 1)[1]
                if tag.startswith("phase:"):
                    meta["phase"] = tag.split("phase:", 1)[1]
            refs = [tag.split("ref:", 1)[1] for tag in tag_list if tag.startswith("ref:")]
            if refs:
                meta["refs"] = refs
            results.append(
                {
                    "ts": row["created_at"],
                    "kind": row["kind"],
                    "summary": text,
                    **meta,
                }
            )
            if len(results) >= k:
                break
        self._log_memory_event(
            "mem_recall_search",
            "recall",
            branch_id=branch_id,
            details={"query": query, "k": k, "results": len(results)},
        )
        return results

    def mem_archival_write(
        self,
        text: str,
        *,
        tags: list[str],
        meta: dict | None = None,
    ) -> str:
        meta_in = meta
        meta = dict(meta or {})
        if self.run_id and not meta.get("run_id"):
            meta["run_id"] = self.run_id
        branch_id = None
        if meta.get("branch_id"):
            branch_id = self._resolve_branch_id(meta["branch_id"]) or meta["branch_id"]
        node_id = meta.get("node_id")
        if not branch_id and node_id:
            branch_id = self._resolve_branch_id(node_id)
        if not branch_id:
            branch_id = self._default_branch_id()
        if not branch_id:
            raise ValueError("mem_archival_write requires a branch id")
        tag_list = list(tags or [])
        if meta.get("run_id"):
            tag_list.append(f"run_id:{meta['run_id']}")
        if node_id:
            tag_list.append(f"node_id:{node_id}")
        record_id = self._insert_archival(
            branch_id,
            text,
            tags=tag_list,
            log_event=False,
        )
        text_str = str(text)
        self._log_memory_event(
            "mem_archival_write",
            "archival",
            branch_id=branch_id,
            node_id=node_id,
            phase=meta.get("phase"),
            details={
                "record_id": record_id,
                "tags": tag_list,
                "text_preview": _truncate(
                    _redact(text_str, self.redact_secrets),
                    self.memory_log_max_chars,
                ),
                "text_chars": len(text_str),
            },
        )
        return str(record_id) if record_id is not None else ""

    def mem_archival_update(
        self,
        record_id: str,
        *,
        text: str | None = None,
        tags: list[str] | None = None,
        meta: dict | None = None,
    ) -> None:
        if not record_id:
            return
        meta = dict(meta or {})
        updates: list[str] = []
        params: list[Any] = []
        if text is not None:
            payload = _redact(str(text), self.redact_secrets)
            updates.append("text=?")
            params.append(payload)
        if tags is not None or meta_in is not None:
            tag_list = list(tags or [])
            if self.run_id and not meta.get("run_id"):
                meta["run_id"] = self.run_id
            if meta.get("run_id"):
                tag_list.append(f"run_id:{meta['run_id']}")
            if meta.get("node_id"):
                tag_list.append(f"node_id:{meta['node_id']}")
            updates.append("tags=?")
            params.append(json.dumps(_normalize_tags(tag_list)))
        if not updates:
            return
        params.append(record_id)
        self._conn.execute(
            f"UPDATE archival SET {', '.join(updates)} WHERE id=?",
            params,
        )
        if self._fts_enabled and text is not None:
            self._conn.execute(
                "UPDATE archival_fts SET text=? WHERE rowid=?",
                (payload, record_id),
            )
        self._conn.commit()
        details = {"record_id": record_id}
        if tags is not None:
            details["tags"] = _normalize_tags(tags)
        if text is not None:
            text_str = str(text)
            details["text_preview"] = _truncate(
                _redact(text_str, self.redact_secrets),
                self.memory_log_max_chars,
            )
            details["text_chars"] = len(text_str)
        self._log_memory_event(
            "mem_archival_update",
            "archival",
            branch_id=meta.get("branch_id") if isinstance(meta, dict) else None,
            node_id=meta.get("node_id") if isinstance(meta, dict) else None,
            phase=meta.get("phase") if isinstance(meta, dict) else None,
            details=details,
        )

    def mem_archival_search(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        k: int = 10,
    ) -> list[dict]:
        branch_id = self._default_branch_id()
        if not branch_id:
            return []
        rows = self.retrieve_archival(
            branch_id=branch_id,
            query=query,
            k=k,
            include_ancestors=True,
            tags_filter=tags,
            log_event=False,
        )
        self._log_memory_event(
            "mem_archival_search",
            "archival",
            branch_id=branch_id,
            details={"query": query, "k": k, "results": len(rows), "tags": tags or []},
        )
        return rows

    def mem_archival_get(self, record_id: str) -> dict:
        row = self._conn.execute(
            "SELECT id, branch_id, text, tags, created_at FROM archival WHERE id=?",
            (record_id,),
        ).fetchone()
        result = dict(row) if row else {}
        self._log_memory_event(
            "mem_archival_get",
            "archival",
            branch_id=result.get("branch_id") if result else None,
            details={
                "record_id": record_id,
                "found": bool(result),
            },
        )
        return result

    def mem_node_fork(self, parent_node_id: str | None, child_node_id: str) -> None:
        parent_branch = self._resolve_branch_id(parent_node_id) if parent_node_id else None
        branch_id = child_node_id or uuid.uuid4().hex
        self.create_branch(parent_branch, node_uid=child_node_id, branch_id=branch_id)
        self._log_memory_event(
            "mem_node_fork",
            "node",
            branch_id=branch_id,
            node_id=child_node_id,
            details={
                "parent_node_id": parent_node_id,
                "parent_branch_id": parent_branch,
                "child_branch_id": branch_id,
            },
        )

    def mem_node_read(self, node_id: str, scope: str = "all") -> dict:
        branch_id = self._resolve_branch_id(node_id)
        if not branch_id:
            return {}
        data: dict[str, Any] = {}
        branch_chain = self._branch_chain(branch_id)
        if scope in ("all", "core"):
            core_rows = self._conn.execute(
                f"SELECT branch_id, key, value, updated_at FROM core_kv WHERE branch_id IN ({','.join(['?']*len(branch_chain))})",
                branch_chain,
            ).fetchall()
            core_latest: dict[str, tuple[str, float]] = {}
            for row in core_rows:
                key = row["key"]
                updated_at = row["updated_at"] or 0
                if key not in core_latest or updated_at > core_latest[key][1]:
                    core_latest[key] = (row["value"], updated_at)
            data["core"] = {key: value for key, (value, _) in core_latest.items()}
        if scope in ("all", "recall"):
            recall_rows = self._fetch_events(branch_chain, self.recall_max_events)
            recall: list[dict[str, Any]] = []
            for row in recall_rows:
                recall.append(
                    {
                        "ts": row["created_at"],
                        "kind": row["kind"],
                        "summary": row["text"],
                        "tags": row["tags"],
                    }
                )
            data["recall"] = recall
        if scope in ("all", "archival"):
            data["archival"] = self.retrieve_archival(
                branch_id=branch_id,
                query="",
                k=self.retrieval_k,
                include_ancestors=True,
                log_event=False,
            )
        self._log_memory_event(
            "mem_node_read",
            "node",
            branch_id=branch_id,
            node_id=node_id,
            details={
                "scope": scope,
                "core_count": len(data.get("core", {}) or {}),
                "recall_count": len(data.get("recall", []) or []),
                "archival_count": len(data.get("archival", []) or []),
            },
        )
        return data

    def mem_node_write(
        self,
        node_id: str,
        *,
        core_updates: dict | None = None,
        recall_event: dict | None = None,
        archival_records: list[dict] | None = None,
    ) -> None:
        branch_id = self._resolve_branch_id(node_id) or node_id
        self._log_memory_event(
            "mem_node_write",
            "node",
            branch_id=branch_id,
            node_id=node_id,
            phase=recall_event.get("phase") if isinstance(recall_event, dict) else None,
            details={
                "core_updates": list(core_updates.keys()) if core_updates else [],
                "recall_event": bool(recall_event),
                "archival_records": len(archival_records or []),
            },
        )
        if core_updates:
            for key, value in core_updates.items():
                ttl = None
                importance = 3
                actual_value = value
                if isinstance(value, dict) and "value" in value:
                    actual_value = value.get("value")
                    ttl = value.get("ttl")
                    importance = _coerce_importance(value.get("importance", 3))
                self.set_core(branch_id, key, str(actual_value), ttl=ttl, importance=importance)
        if recall_event:
            event = dict(recall_event)
            if "node_id" not in event:
                event["node_id"] = node_id
            self.mem_recall_append(event)
        if archival_records:
            for record in archival_records:
                if not isinstance(record, dict):
                    continue
                text = record.get("text", "")
                tags = record.get("tags", [])
                meta = record.get("meta", {})
                if "node_id" not in meta:
                    meta = {**meta, "node_id": node_id}
                self.mem_archival_write(str(text), tags=list(tags or []), meta=meta)

    def mem_node_promote(self, child_node_id: str, parent_node_id: str, policy: str) -> None:
        child_branch = self._resolve_branch_id(child_node_id)
        parent_branch = self._resolve_branch_id(parent_node_id)
        if not child_branch or not parent_branch:
            return
        policy = str(policy or "").strip()
        self._log_memory_event(
            "mem_node_promote",
            "node",
            branch_id=parent_branch,
            node_id=parent_node_id,
            details={"policy": policy, "child_node_id": child_node_id},
        )
        if policy == "resources_update":
            for key in (RESOURCE_INDEX_KEY, RESOURCE_INDEX_JSON_KEY, RESOURCE_DIGEST_KEY, RESOURCE_USED_KEY):
                value = self.get_core(child_branch, key, log_event=False)
                if value:
                    self.set_core(
                        parent_branch,
                        key,
                        value,
                        importance=5,
                        log_event=False,
                    )
            return
        if policy == "writeup_ready":
            run_dir = self.workspace_root or (self.db_path.parent.parent if self.db_path else Path("."))
            try:
                self.generate_final_memory_for_paper(
                    run_dir=run_dir,
                    root_branch_id=parent_branch,
                    best_branch_id=child_branch,
                    artifacts_index={"promoted_from": child_node_id},
                )
            except Exception:
                pass
            return

        if policy != "selected_best":
            return
        child_rows = self._conn.execute(
            "SELECT key, value FROM core_kv WHERE branch_id=?",
            (child_branch,),
        ).fetchall()
        meta_rows = self._conn.execute(
            "SELECT key, importance FROM core_meta WHERE branch_id=?",
            (child_branch,),
        ).fetchall()
        importance_map = {row["key"]: _coerce_importance(row["importance"]) for row in meta_rows}
        keyword_keys = {"env", "condition", "result", "metric", "implementation", "failure"}
        for row in child_rows:
            key = row["key"]
            value = row["value"]
            importance = importance_map.get(key, 3)
            matches = any(token in key.lower() for token in keyword_keys)
            if importance < 4 and not matches:
                continue
            text = f"Promoted from {child_node_id}\n{key}: {value}"
            record_id = self.mem_archival_write(
                text,
                tags=["PROMOTED", f"policy:{policy}", f"source_node:{child_node_id}"],
                meta={"node_id": parent_node_id},
            )
            if record_id:
                self.set_core(
                    parent_branch,
                    key,
                    f"ref:{record_id}",
                    importance=max(importance, 4),
                )

    def mem_resources_index_update(self, run_id: str, index_text: str) -> None:
        branch_id = self._default_branch_id()
        if not branch_id:
            return
        if run_id and not self.run_id:
            self.run_id = str(run_id)
        self.set_core(
            branch_id,
            RESOURCE_INDEX_KEY,
            index_text,
            importance=5,
            log_event=False,
        )
        self._log_memory_event(
            "mem_resources_index_update",
            "resources",
            branch_id=branch_id,
            details={
                "run_id": run_id,
                "index_chars": len(index_text or ""),
            },
        )

    def mem_resources_snapshot_upsert(self, run_id: str, item_name: str, snapshot: dict) -> None:
        tags = ["RESOURCE_ITEM", f"resource_name:{item_name}"]
        if run_id:
            tags.append(f"run_id:{run_id}")
        payload = json.dumps(snapshot, ensure_ascii=True, indent=2)
        self.mem_archival_write(payload, tags=tags, meta={"run_id": run_id})
        self._log_memory_event(
            "mem_resources_snapshot_upsert",
            "resources",
            branch_id=self._default_branch_id(),
            details={
                "run_id": run_id,
                "item_name": item_name,
                "payload_chars": len(payload),
            },
        )

    def mem_resources_resolve_and_refresh(self, run_id: str) -> None:
        branch_id = self._default_branch_id()
        if not branch_id:
            return
        raw_index = self.get_core(branch_id, RESOURCE_INDEX_JSON_KEY, log_event=False)
        resource_file = None
        if raw_index:
            try:
                parsed = json.loads(raw_index)
                resource_file = parsed.get("resource_file")
            except json.JSONDecodeError:
                resource_file = None
        if not resource_file:
            self._log_memory_event(
                "mem_resources_resolve_and_refresh",
                "resources",
                branch_id=branch_id,
                details={"run_id": run_id, "resource_file": "", "status": "skipped"},
            )
            return
        if run_id and not self.run_id:
            self.run_id = str(run_id)
        workspace_root = self.workspace_root or self.db_path.parent.parent
        self._log_memory_event(
            "mem_resources_resolve_and_refresh",
            "resources",
            branch_id=branch_id,
            details={"run_id": run_id, "resource_file": resource_file, "status": "refresh"},
        )
        snapshot = build_resource_snapshot(
            resource_file,
            workspace_root=workspace_root,
            ai_scientist_root=str(self.ai_scientist_root) if self.ai_scientist_root else None,
            phase_mode=_cfg_get(self.config, "phase_mode", "unknown"),
            log=logger,
        )
        update_resource_snapshot_if_changed(snapshot, self)

    def _render_resource_items(self, branch_id: str, task_hint: str | None) -> str:
        query = (task_hint or "").strip()
        if not query:
            query = "resource"
        rows = self.retrieve_archival(
            branch_id=branch_id,
            query=query,
            k=self.retrieval_k,
            include_ancestors=True,
            tags_filter=[RESOURCE_ITEM_TAG],
            log_event=False,
        )
        if not rows:
            return ""
        resource_ids: set[str] = set()
        chunks: list[str] = []
        for row in rows:
            for rid in _extract_resource_ids(row.get("tags")):
                if rid:
                    resource_ids.add(rid)
            chunks.append(_truncate(row.get("text", ""), 1200))
        if resource_ids:
            note = f"prompt:{query}"
            for rid in sorted(resource_ids):
                track_resource_usage(rid, {"ltm": self, "branch_id": branch_id, "note": note})
        return "\n\n---\n\n".join(chunk for chunk in chunks if chunk).strip()

    def render_for_prompt(
        self,
        branch_id: str,
        task_hint: str | None,
        budget_chars: int,
    ) -> str:
        if not branch_id:
            return ""
        branch_ids = self._branch_chain(branch_id)
        if not branch_ids:
            return ""

        core_rows = self._conn.execute(
            f"SELECT branch_id, key, value, updated_at FROM core_kv WHERE branch_id IN ({','.join(['?']*len(branch_ids))})",
            branch_ids,
        ).fetchall()
        core_latest: dict[str, tuple[str, float]] = {}
        for row in core_rows:
            key = row["key"]
            updated_at = row["updated_at"] or 0
            if key not in core_latest or updated_at > core_latest[key][1]:
                core_latest[key] = (row["value"], updated_at)

        core_lines: list[str] = []
        core_latest.pop("idea_md_hash", None)
        resource_index = core_latest.pop(RESOURCE_INDEX_KEY, (None, 0))[0]
        core_latest.pop(RESOURCE_INDEX_JSON_KEY, None)
        core_latest.pop(RESOURCE_DIGEST_KEY, None)
        core_latest.pop(RESOURCE_USED_KEY, None)
        idea_summary = core_latest.pop("idea_md_summary", (None, 0))[0]
        if self.always_inject_idea_summary:
            core_lines.append(
                f"- Idea summary: {idea_summary or '(not available)'}"
            )
        phase0_summary = core_latest.pop("phase0_summary", (None, 0))[0]
        if self.always_inject_phase0_summary:
            core_lines.append(
                f"- Phase 0 internal summary: {phase0_summary or '(not available)'}"
            )

        for key, (value, _) in sorted(core_latest.items(), key=lambda kv: kv[0]):
            core_lines.append(f"- {key}: {value}")
        core_text = "\n".join(core_lines).strip()

        resource_items_text = self._render_resource_items(branch_id, task_hint)

        recall_rows = self._fetch_events(branch_ids, self.recall_max_events)
        recall_lines = []
        for row in recall_rows:
            tags = ""
            try:
                tag_list = json.loads(row["tags"]) if row["tags"] else []
                if tag_list:
                    tags = f" (tags: {', '.join(tag_list)})"
            except json.JSONDecodeError:
                tags = ""
            recall_lines.append(f"- [{row['kind']}] {row['text']}{tags}")
        recall_text = "\n".join(recall_lines).strip()

        archival_rows = self.retrieve_archival(
            branch_id=branch_id,
            query=task_hint or "",
            k=self.retrieval_k,
            include_ancestors=True,
            log_event=False,
        )
        archival_lines = []
        for row in archival_rows:
            tags = ""
            try:
                tag_list = json.loads(row.get("tags") or "[]")
                if tag_list:
                    tags = f" (tags: {', '.join(tag_list)})"
            except json.JSONDecodeError:
                tags = ""
            snippet = _truncate(row.get("text", ""), 400)
            archival_lines.append(f"- {snippet}{tags}")
        archival_text = "\n".join(archival_lines).strip()

        sections = []
        if resource_index:
            sections.append(("Resource Index", resource_index))
        if core_text:
            sections.append(("Core Memory", core_text))
        if recall_text:
            sections.append(("Recall Memory", recall_text))
        if archival_text:
            sections.append(("Archival Memory", archival_text))
        if resource_items_text:
            sections.append(("Resource Memory", resource_items_text))

        if not sections:
            self._log_memory_event(
                "render_for_prompt",
                "prompt",
                branch_id=branch_id,
                phase=task_hint,
                details={
                    "budget_chars": budget_chars,
                    "core_count": len(core_lines),
                    "recall_count": len(recall_rows),
                    "archival_count": len(archival_rows),
                    "resource_items": 0,
                },
            )
            return ""

        remaining = max(int(budget_chars), 0)
        rendered_parts: list[str] = []
        for title, body in sections:
            block = f"{title}:\n{body}\n"
            if remaining <= 0:
                break
            if len(block) <= remaining:
                rendered_parts.append(block)
                remaining -= len(block)
            else:
                if remaining > len(title) + 10:
                    truncated = _truncate(block, remaining)
                    rendered_parts.append(truncated)
                break
        rendered = "\n".join(part.strip() for part in rendered_parts if part.strip()).strip()
        self._log_memory_event(
            "render_for_prompt",
            "prompt",
            branch_id=branch_id,
            phase=task_hint,
            details={
                "budget_chars": budget_chars,
                "core_count": len(core_lines),
                "recall_count": len(recall_rows),
                "archival_count": len(archival_rows),
                "resource_items": 1 if resource_items_text else 0,
            },
        )
        return rendered

    def ingest_phase0_internal_info(
        self,
        branch_id: str,
        node_uid: str,
        phase0_payload: Any,
        artifact_paths: Sequence[str] | None,
        command_str: str | None,
    ) -> None:
        memory_dir = self.db_path.parent
        payload_obj = phase0_payload
        if is_dataclass(phase0_payload):
            payload_obj = asdict(phase0_payload)
        artifacts = []
        for path in artifact_paths or []:
            p = Path(path)
            if not p.exists():
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                content = ""
            artifacts.append(
                {
                    "path": str(p),
                    "content": _truncate(content, 5000),
                }
            )

        payload = {
            "branch_id": branch_id,
            "node_uid": node_uid,
            "command": command_str,
            "phase0_payload": payload_obj,
            "artifacts": artifacts,
            "created_at": _now_ts(),
        }
        json_path = memory_dir / "phase0_internal_info.json"
        md_path = memory_dir / "phase0_internal_info.md"
        try:
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write phase0 internal json: %s", exc)

        md_lines = [
            "# Phase 0 Internal Info",
            "",
            f"- node_uid: {node_uid}",
            f"- branch_id: {branch_id}",
            f"- command: {command_str or ''}",
            "",
            "## Phase 0 Payload",
            "```json",
            json.dumps(payload_obj, indent=2),
            "```",
        ]
        if artifacts:
            md_lines.append("")
            md_lines.append("## Artifacts")
            for item in artifacts:
                md_lines.append(f"### {item['path']}")
                md_lines.append("```")
                md_lines.append(item["content"])
                md_lines.append("```")
        md_text = "\n".join(md_lines)
        try:
            md_path.write_text(md_text, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write phase0 internal md: %s", exc)

        summary = _summarize_phase0(payload_obj, command_str)
        self.set_core(
            branch_id,
            "phase0_summary",
            summary,
            importance=5,
            op_name="ingest_phase0_internal_info",
            phase="phase0",
            node_id=node_uid,
        )
        archival_text = f"{summary}\n\n{md_text}"
        self.mem_archival_write(
            archival_text,
            tags=["PHASE0_INTERNAL", f"node_uid:{node_uid}", f"branch_id:{branch_id}"],
            meta={
                "node_id": node_uid,
                "run_id": self.run_id,
                "branch_id": branch_id,
                "phase": "phase0",
            },
        )

    def ingest_idea_md(
        self,
        branch_id: str,
        node_uid: str,
        idea_path: str | Path,
        is_root: bool = False,
    ) -> None:
        path = Path(idea_path)
        if not path.exists():
            return
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read idea.md: %s", exc)
            return
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        previous_hash = self.get_core(branch_id, "idea_md_hash", log_event=False)
        if previous_hash == content_hash:
            return

        summary = _summarize_idea(content)
        self.set_core(
            branch_id,
            "idea_md_summary",
            summary,
            importance=5,
            op_name="ingest_idea_md",
            phase="idea_md",
            node_id=node_uid,
        )
        self.set_core(
            branch_id,
            "idea_md_hash",
            content_hash,
            op_name="ingest_idea_md",
            phase="idea_md",
            node_id=node_uid,
        )
        tags = ["IDEA_MD", f"node_uid:{node_uid}", f"branch_id:{branch_id}"]
        if is_root:
            tags.append("ROOT_IDEA")
        archival_text = f"{summary}\n\n{content}"
        self.mem_archival_write(
            archival_text,
            tags=tags,
            meta={
                "node_id": node_uid,
                "run_id": self.run_id,
                "branch_id": branch_id,
                "phase": "idea_md",
            },
        )

    def generate_final_memory_for_paper(
        self,
        run_dir: str | Path,
        root_branch_id: str,
        best_branch_id: str | None,
        artifacts_index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(run_dir)
        memory_dir = run_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        best_branch = best_branch_id or root_branch_id
        core_snapshot = self.render_for_prompt(best_branch, task_hint="", budget_chars=2000)
        idea_archival = self.retrieve_archival(
            best_branch, query="idea", k=1, include_ancestors=True, tags_filter=["IDEA_MD"]
        )
        idea_text = idea_archival[0]["text"] if idea_archival else ""
        idea_summary = self.get_core(best_branch, "idea_md_summary") or ""
        if not idea_summary and idea_text:
            idea_summary = _summarize_idea(idea_text)
        phase0_summary = self.get_core(best_branch, "phase0_summary") or ""
        failure_events = self._fetch_events(self._branch_chain(best_branch), self.recall_max_events)
        failure_notes = [
            row["text"]
            for row in failure_events
            if str(row["kind"]).lower() in {"error", "exception", "failure"}
        ]

        resource_index = {}
        raw_index = self.get_core(root_branch_id, RESOURCE_INDEX_JSON_KEY)
        if not raw_index and best_branch:
            raw_index = self.get_core(best_branch, RESOURCE_INDEX_JSON_KEY)
        if raw_index:
            try:
                resource_index = json.loads(raw_index)
            except json.JSONDecodeError:
                resource_index = {}
        item_index = {
            item.get("id"): item
            for item in resource_index.get("items", [])
            if isinstance(item, dict) and item.get("id")
        }
        used_ids: list[str] = []
        raw_used = self.get_core(root_branch_id, RESOURCE_USED_KEY)
        if not raw_used and best_branch:
            raw_used = self.get_core(best_branch, RESOURCE_USED_KEY)
        if raw_used:
            try:
                parsed = json.loads(raw_used)
                if isinstance(parsed, list):
                    used_ids = [str(item) for item in parsed]
            except json.JSONDecodeError:
                used_ids = []
        usage_notes: dict[str, list[str]] = {}
        event_rows = self._fetch_events(self._branch_chain(best_branch), 200)
        for row in event_rows:
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except json.JSONDecodeError:
                tags = []
            if RESOURCE_USED_TAG not in tags and str(row["kind"]).lower() != "resource_used":
                continue
            text = row["text"] or ""
            parts = [part.strip() for part in text.split("|", 1)]
            if not parts:
                continue
            rid = parts[0]
            note = parts[1] if len(parts) > 1 else ""
            if rid:
                usage_notes.setdefault(rid, [])
                if note and note not in usage_notes[rid]:
                    usage_notes[rid].append(note)
                if rid not in used_ids:
                    used_ids.append(rid)

        resources_used: list[dict[str, Any]] = []
        for rid in used_ids:
            item = item_index.get(rid, {})
            if not item:
                resources_used.append({"id": rid, "note": "No index entry available."})
                continue
            summary = ""
            if item.get("class") in {"template", "document"}:
                rows = self.retrieve_archival(
                    branch_id=root_branch_id or best_branch,
                    query=rid,
                    k=1,
                    include_ancestors=True,
                    tags_filter=[f"resource_id:{rid}"],
                )
                if rows:
                    summary = _truncate(_strip_frontmatter(rows[0].get("text", "")), 300)
            resources_used.append(
                {
                    "id": rid,
                    "class": item.get("class", ""),
                    "name": item.get("name", ""),
                    "source": item.get("source", ""),
                    "resource": item.get("resource", ""),
                    "digest": item.get("digest", ""),
                    "staged_path": item.get("staged_path", ""),
                    "dest_path": item.get("dest_path", item.get("container_path", "")),
                    "availability": item.get("availability", ""),
                    "summary": summary,
                    "usage_notes": usage_notes.get(rid, []),
                }
            )

        sections = {
            "title_candidates": _truncate(idea_summary, 300),
            "abstract_material": _truncate(idea_summary, 500),
            "problem_statement": _truncate(idea_summary, 400),
            "hypothesis": _truncate(idea_summary, 300),
            "method": _truncate(idea_summary, 400),
            "experimental_setup": _truncate(phase0_summary, 400),
            "phase0_internal_info_summary": _truncate(phase0_summary, 400),
            "results": _truncate(core_snapshot, 600),
            "ablations_negative": "No explicit ablation notes captured.",
            "failure_modes_timeline": _truncate("\n".join(failure_notes) or "No failures recorded.", 600),
            "threats_to_validity": "OS noise, NUMA placement, container overhead, measurement jitter.",
            "reproducibility_checklist": "Config file, run commands, artifact paths, random seeds.",
            "narrative_bullets": "Related work positioning, key trade-offs, implications.",
            "resources_used": resources_used,
        }
        if artifacts_index:
            sections["artifacts_index"] = artifacts_index

        # Build final writeup memory payload (condensed but reproducible)
        phase0_path = memory_dir / "phase0_internal_info.json"
        phase0_payload = {}
        if phase0_path.exists():
            try:
                phase0_payload = json.loads(phase0_path.read_text(encoding="utf-8"))
            except Exception:
                phase0_payload = {}
        idea_path = run_dir / "idea.md"
        if not idea_path.exists():
            idea_path = Path(_cfg_get(self.config, "desc_file", "")) if _cfg_get(self.config, "desc_file", "") else idea_path
        resource_snapshot_path = memory_dir / "resource_snapshot.json"
        resource_snapshot = {}
        if resource_snapshot_path.exists():
            try:
                resource_snapshot = json.loads(resource_snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                resource_snapshot = {}
        resources_section = {
            "resource_file": {
                "path": resource_snapshot.get("resource_file") or resource_index.get("resource_file", ""),
                "sha": resource_snapshot.get("resource_file_sha") or resource_index.get("resource_file_sha", ""),
                "digest": resource_index.get("resource_digest", ""),
            },
            "normalized": resource_snapshot.get("normalized") or resource_index.get("normalized", {}),
            "items": resource_snapshot.get("items") or resource_index.get("items", []),
            "used": resources_used,
        }
        branch_chain = self._branch_chain(best_branch)
        provenance = []
        for bid in branch_chain:
            row = self._conn.execute(
                "SELECT node_uid FROM branches WHERE id=?", (bid,)
            ).fetchone()
            if row and row["node_uid"]:
                provenance.append(row["node_uid"])
            else:
                provenance.append(bid)
        method_changes = []
        results_notes = []
        for row in event_rows:
            kind = str(row["kind"]).lower()
            if kind == "node_created":
                method_changes.append(_truncate(row["text"] or "", 800))
            if kind == "node_result":
                results_notes.append(_truncate(row["text"] or "", 800))
        writeup_memory = {
            "run_id": self.run_id or run_dir.name,
            "idea": {
                "summary": idea_summary,
                "path": str(idea_path) if idea_path and idea_path.exists() else "",
                "content": _truncate(idea_text or "", 4000),
            },
            "phase0_env": {
                "summary": phase0_summary,
                "path": str(phase0_path) if phase0_path.exists() else "",
                "raw": phase0_payload,
            },
            "resources": resources_section,
            "method_changes": method_changes,
            "experiments": {
                "phase0_plan": phase0_payload.get("phase0_payload", {}),
                "phase0_command": phase0_payload.get("command", ""),
                "workspace_dir": str(run_dir),
            },
            "results": {
                "core_snapshot": core_snapshot,
                "notes": results_notes,
                "artifacts": artifacts_index or {},
            },
            "negative_results": failure_notes,
            "provenance": provenance,
        }
        writeup_path = memory_dir / "final_writeup_memory.json"
        writeup_path.write_text(json.dumps(writeup_memory, indent=2), encoding="utf-8")

        md_sections = [
            "# Final Memory For Paper",
            "",
            "## Title Candidates / Abstract Material",
            sections["title_candidates"],
            "",
            "## Problem Statement / Motivation",
            sections["problem_statement"],
            "",
            "## Hypothesis",
            sections["hypothesis"],
            "",
            "## Method",
            sections["method"],
            "",
            "## Experimental Setup",
            sections["experimental_setup"],
            "",
            "## Phase0 Internal Info Summary",
            sections["phase0_internal_info_summary"],
            "",
            "## Results",
            sections["results"],
            "",
            "## Ablations / Negative Results",
            sections["ablations_negative"],
            "",
            "## Failure Modes & Debugging Timeline",
            sections["failure_modes_timeline"],
            "",
            "## Threats to Validity",
            sections["threats_to_validity"],
            "",
            "## Reproducibility Checklist",
            sections["reproducibility_checklist"],
            "",
            "## Narrative Bullets",
            sections["narrative_bullets"],
            "",
            "## Resources Used",
            "",
        ]
        if resources_used:
            for entry in resources_used:
                label = f"{entry.get('class')}/{entry.get('name')}".strip("/")
                src = entry.get("source") or "unknown"
                digest = entry.get("digest") or ""
                staged = entry.get("staged_path") or ""
                dest = entry.get("dest_path") or ""
                md_sections.append(f"- {label} ({src}) digest={digest} staged={staged} dest={dest}")
                summary = entry.get("summary") or ""
                if summary:
                    md_sections.append(f"  Summary: {summary}")
                notes = entry.get("usage_notes") or []
                if notes:
                    md_sections.append(f"  Usage: {', '.join(notes)}")
            md_sections.append("")
        else:
            md_sections.append("No explicit resource usage recorded.")
            md_sections.append("")
        md_path = memory_dir / str(_cfg_get(self.config, "final_memory_filename_md", "final_memory_for_paper.md"))
        json_path = memory_dir / str(_cfg_get(self.config, "final_memory_filename_json", "final_memory_for_paper.json"))
        md_path.write_text("\n".join(md_sections), encoding="utf-8")
        json_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")
        return sections
