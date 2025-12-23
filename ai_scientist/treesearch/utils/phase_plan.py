import json
from pathlib import Path
from typing import Any, Dict, List


class PhasePlanError(ValueError):
    """Raised when the LLM phase plan is missing required structure."""


def _strip_json_wrappers(raw_text: str) -> str:
    """Remove markdown fences and keep the JSON portion."""
    cleaned = raw_text.strip()
    if "```" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def extract_phase_artifacts(raw_text: str, *, default_language: str = "c") -> Dict[str, Any]:
    """Parse and validate the JSON returned by the LLM for split-phase execution."""
    def normalize_language(value: str | None) -> str:
        lang = str(value or "").strip().lower()
        if not lang:
            return "c"
        if lang in {"c++", "cxx"}:
            return "cpp"
        return lang

    def placeholder_source(lang: str) -> tuple[str, str]:
        if lang == "cpp":
            return (
                "src/main.cpp",
                (
                    "// Auto-generated fallback to keep pipeline alive\n"
                    "int main() {\n"
                    "    return 0;\n"
                    "}\n"
                ),
            )
        if lang == "python":
            return (
                "src/main.py",
                (
                    "# Auto-generated fallback to keep pipeline alive\n"
                    "print('Placeholder code; LLM omitted files')\n"
                ),
            )
        return (
            "src/main.c",
            (
                "/* Auto-generated fallback to keep pipeline alive */\n"
                "int main(void) {\n"
                "    return 0;\n"
                "}\n"
            ),
        )

    default_language = normalize_language(default_language)
    cleaned = _strip_json_wrappers(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            import ast

            parsed = ast.literal_eval(cleaned)
        except Exception as exc:  # pragma: no cover - defensive path
            raise PhasePlanError(f"Failed to parse LLM JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise PhasePlanError("LLM response must be a JSON object.")

    phase_payload = parsed.get("phase_artifacts", parsed.get("plan"))
    if phase_payload is None:
        phase_keys = {"download", "coding", "compile", "run"}
        if isinstance(parsed, dict) and phase_keys.issubset(parsed.keys()):
            phase_payload = parsed
        else:
            raise PhasePlanError("phase_artifacts is missing from LLM response.")

    if isinstance(phase_payload, list) and phase_payload:
        phase_payload = phase_payload[0]
    if not isinstance(phase_payload, dict):
        raise PhasePlanError("phase_artifacts must be an object.")

    # Normalize phase sections
    phase: Dict[str, Any] = {}
    for key, default in (
        ("download", {"commands": [], "notes": "fill download/install commands"}),
        ("coding", {"workspace": {}}),
        ("compile", {"build_plan": {}}),
        ("run", {"commands": [], "expected_outputs": [], "notes": "fill run commands"}),
    ):
        section = phase_payload.get(key, default)
        if isinstance(section, list) and section:
            section = section[0]
        phase[key] = section if isinstance(section, dict) else default

    coding = phase["coding"]
    workspace = coding.get("workspace", {})
    if not isinstance(workspace, dict) or not workspace.get("files"):
        placeholder_path, placeholder_content = placeholder_source(default_language)
        workspace = {
            "root": "/workspace",
            "tree": ["workspace/", "workspace/src/", "workspace/working/"],
            "files": [
                {
                    "path": placeholder_path,
                    "mode": "0644",
                    "encoding": "utf-8",
                    "content": placeholder_content,
                }
            ],
        }
    workspace.setdefault("root", "/workspace")
    workspace.setdefault("tree", ["workspace/", "workspace/working/"])
    coding["workspace"] = workspace

    compile_section = phase["compile"]
    build_plan = compile_section.get("build_plan") or {}
    if isinstance(build_plan, list) and build_plan:
        build_plan = build_plan[0]
    if not isinstance(build_plan, dict):
        build_plan = {}
    build_plan.setdefault("language", default_language)
    build_plan.setdefault("compiler_selected", "")
    build_plan.setdefault("cflags", [])
    build_plan.setdefault("ldflags", [])
    build_plan.setdefault("workdir", "/workspace")
    if default_language == "python":
        build_plan.setdefault("output", "working/experiment_data.npy")
    else:
        build_plan.setdefault("output", "bin/a.out")
    compile_section["build_plan"] = build_plan

    if not build_plan.get("compiler_selected"):
        raise PhasePlanError("build_plan.compiler_selected is required.")

    run_section = phase["run"]
    expected_outputs = run_section.get("expected_outputs")
    if not expected_outputs:
        run_section["expected_outputs"] = ["working/experiment_data.npy"]

    constraints = parsed.get("constraints")
    if isinstance(constraints, list) and constraints:
        constraints = constraints[0]
    if not isinstance(constraints, dict):
        constraints = {
            "allow_sudo_in_singularity": True,
            "allow_apt_get_in_singularity": True,
            "write_only_under_workspace": True,
            "no_absolute_paths": True,
            "no_parent_traversal": True,
            "python_output_must_use_numpy": True,
            "non_python_output_must_use_cnpy": True,
        }

    parsed["phase_artifacts"] = phase
    parsed["constraints"] = constraints
    return parsed


def _validate_relative_path(base: Path, relative_path: str) -> Path:
    if relative_path.startswith("/"):
        raise PhasePlanError(f"Absolute paths are not allowed: {relative_path}")
    dest = (base / relative_path).resolve()
    try:
        dest.relative_to(base)
    except ValueError as exc:
        raise PhasePlanError(f"Path escapes workspace: {relative_path}") from exc
    return dest


def apply_workspace_plan(workspace_root: Path, workspace_plan: Dict[str, Any], *, expected_root: str) -> List[Path]:
    """Create files described by coding.workspace.files under workspace_root."""
    actual_root = workspace_plan.get("root")
    if actual_root:
        normalized_actual = str(actual_root).rstrip("/")
        normalized_expected = str(expected_root).rstrip("/")
    else:
        normalized_actual = None
        normalized_expected = str(expected_root).rstrip("/")
    if normalized_actual and normalized_actual != normalized_expected:
        raise PhasePlanError(
            f"Workspace root must be '{expected_root}', got '{actual_root}'."
        )
    created: List[Path] = []
    root_prefix = expected_root.lstrip("/")
    for tree_entry in workspace_plan.get("tree", []):
        rel_entry = tree_entry.lstrip("/").rstrip("/")
        if root_prefix and rel_entry.startswith(root_prefix):
            rel_entry = rel_entry[len(root_prefix) :].lstrip("/")
        if not rel_entry:
            continue
        tree_path = _validate_relative_path(workspace_root, rel_entry)
        tree_path.mkdir(parents=True, exist_ok=True)
    files = workspace_plan.get("files", [])
    for file_entry in files:
        path = _validate_relative_path(workspace_root, file_entry["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = file_entry.get("content", "")
        encoding = file_entry.get("encoding", "utf-8")
        mode = file_entry.get("mode")
        path.write_text(content, encoding=encoding)
        if mode:
            try:
                path.chmod(int(mode, 8))
            except ValueError:
                pass
        created.append(path)
    return created


def combine_sources_for_display(files: List[Dict[str, Any]]) -> str:
    """Build a readable string representation of generated files."""
    parts: list[str] = []
    for entry in files:
        parts.append(f"// File: {entry.get('path','')}\n{entry.get('content','')}")
    return "\n\n".join(parts)
