import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from ai_scientist.treesearch.utils.response import sanitize_memory_update_tags


class PhasePlanError(ValueError):
    """Raised when the LLM phase plan is missing required structure."""


class MissingMemoryUpdateError(PhasePlanError):
    """Raised when memory is enabled but <memory_update> block is missing."""


def extract_memory_update_block(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract <memory_update> block from text and return (updates, remaining_text).

    Args:
        raw_text: The raw LLM response text.

    Returns:
        A tuple of (memory_updates_dict, text_without_memory_block).
        If no <memory_update> block is found, returns (None, original_text).
    """
    if not raw_text:
        return None, raw_text

    # Sanitize malformed opening tags (e.g., with injected attributes)
    raw_text = sanitize_memory_update_tags(raw_text)

    pattern = r'<memory_update>\s*(.*?)\s*</memory_update>'
    match = re.search(pattern, raw_text, re.DOTALL)

    if not match:
        return None, raw_text

    # Extract the JSON content from the memory_update block
    memory_json = match.group(1).strip()

    # Remove the memory_update block from the text
    remaining_text = re.sub(pattern, '', raw_text, flags=re.DOTALL).strip()

    # Parse the memory update JSON
    try:
        if not memory_json or memory_json == '{}':
            return {}, remaining_text
        updates = json.loads(memory_json)
        if not isinstance(updates, dict):
            return None, raw_text
        return updates, remaining_text
    except json.JSONDecodeError:
        # If JSON parsing fails, return None but still remove the block
        return None, remaining_text


def _strip_json_wrappers(raw_text: str) -> str:
    """Remove markdown fences and keep the JSON portion."""
    cleaned = raw_text.strip()
    if "```" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def extract_phase_artifacts(
    raw_text: str,
    *,
    default_language: str = "c",
    require_memory_update: bool = False,
) -> Dict[str, Any]:
    """Parse and validate the JSON returned by the LLM for split-phase execution.

    Args:
        raw_text: The raw LLM response text.
        default_language: Default programming language for placeholders.
        require_memory_update: If True, raise MissingMemoryUpdateError when
            <memory_update> block is not found.

    Returns:
        A dict containing 'phase_artifacts', 'constraints', and optionally
        'memory_update' if a <memory_update> block was found.

    Raises:
        PhasePlanError: If the JSON structure is invalid.
        MissingMemoryUpdateError: If require_memory_update=True and no
            <memory_update> block is found.
    """
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
        if lang == "fortran":
            return (
                "src/main.f90",
                (
                    "! Auto-generated fallback to keep pipeline alive\n"
                    "program main\n"
                    "    implicit none\n"
                    "    print *, 'Placeholder code; LLM omitted files'\n"
                    "end program main\n"
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

    # First, extract memory_update block if present
    memory_updates, remaining_text = extract_memory_update_block(raw_text)

    # Check if memory_update is required but missing
    if require_memory_update and memory_updates is None:
        raise MissingMemoryUpdateError(
            "Memory is enabled but <memory_update> block is missing from LLM response. "
            "The response must start with a <memory_update>...</memory_update> block."
        )

    default_language = normalize_language(default_language)
    cleaned = _strip_json_wrappers(remaining_text)
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

    # Include memory_update in the result if it was found
    if memory_updates is not None:
        parsed["memory_update"] = memory_updates

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


def _get_lang_for_path(path: str) -> str:
    """Return the appropriate language identifier for code blocks based on file extension."""
    path_lower = path.lower()
    filename = path_lower.split('/')[-1]

    # Python
    if path_lower.endswith('.py'):
        return "python"
    # C
    if path_lower.endswith('.c'):
        return "c"
    # C++
    if path_lower.endswith(('.cpp', '.cc', '.cxx', '.hpp', '.hxx')):
        return "cpp"
    # C/C++ headers
    if path_lower.endswith('.h'):
        return "c"
    # Makefile
    if filename == 'makefile' or filename.startswith('makefile.') or filename.endswith('.mk'):
        return "makefile"
    # Shell
    if path_lower.endswith(('.sh', '.bash', '.zsh')):
        return "bash"
    # YAML
    if path_lower.endswith(('.yaml', '.yml')):
        return "yaml"
    # JSON
    if path_lower.endswith('.json'):
        return "json"
    # Fortran
    if path_lower.endswith(('.f', '.f90', '.f95', '.for')):
        return "fortran"
    # Default
    return ""


def combine_sources_for_display(files: List[Dict[str, Any]]) -> str:
    """Build a readable string representation of generated files (no code block wrappers)."""
    parts: list[str] = []
    for entry in files:
        parts.append(f"// File: {entry.get('path','')}\n{entry.get('content','')}")
    return "\n\n".join(parts)


def wrap_sources_for_display(files: List[Dict[str, Any]]) -> str:
    """Build a readable string with each file wrapped in its own appropriately-typed code block."""
    parts: list[str] = []
    for entry in files:
        path = entry.get('path', '')
        content = entry.get('content', '')
        lang = _get_lang_for_path(path)
        parts.append(f"```{lang}\n// File: {path}\n{content}\n```")
    return "\n\n".join(parts)


def wrap_combined_code(code: str, fallback_lang: str = "") -> str:
    """Parse a combined sources string and rewrap each file in its own appropriately-typed code block.

    This handles the output of combine_sources_for_display by parsing '// File: ...' markers.
    """
    import re
    # Pattern to match: // File: <path>\n<content until next // File: or end>
    pattern = re.compile(r'// File:\s*([^\n]+)\n', re.MULTILINE)

    matches = list(pattern.finditer(code))
    if not matches:
        # No file markers found, wrap the whole code with fallback language
        return f"```{fallback_lang}\n{code}\n```"

    parts: list[str] = []
    for i, match in enumerate(matches):
        path = match.group(1).strip()
        start = match.end()
        # Content goes until the next match or end of string
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(code)
        content = code[start:end].rstrip('\n')
        lang = _get_lang_for_path(path)
        parts.append(f"```{lang}\n// File: {path}\n{content}\n```")

    return "\n\n".join(parts)
