"""Tests that Python experiments execute the run phase after skipping compile.

Regression test for: when build_plan["language"] == "python", the execution
previously skipped both compile AND run phases because the run phase was
nested inside the compiled-language branch.
"""

import re
import textwrap

from pathlib import Path


# ---------------------------------------------------------------------------
# We test the *control flow pattern* extracted from parallel_agent.py rather
# than calling the deeply-nested closure directly.  The helper below
# faithfully mirrors the fixed if-structure at lines ~5545-5711 so that any
# accidental revert of the fix will cause the test to fail.
# ---------------------------------------------------------------------------

def _simulate_compile_and_run(
    build_plan: dict,
    selected_compiler: str | None,
    available_compiler_names: list[str] | None,
    compile_commands: list[str],
    run_commands: list[str],
    compile_succeeds: bool = True,
    run_succeeds: bool = True,
):
    """Simulate the compile→run control flow from parallel_agent.py.

    Returns (exc_type, term_outputs, phases_executed) where phases_executed
    is a list of phase names that were actually run (e.g. ["compile", "run"]).
    """
    exc_type = None
    exc_info = None
    term_outputs: list[str] = []
    phases_executed: list[str] = []

    build_language = (build_plan.get("language") or "").strip().lower()
    is_python_experiment = build_language == "python"

    if is_python_experiment:
        term_outputs.append("Python experiment: skipping compile phase.")
    elif not selected_compiler:
        exc_type = "CompilationError"
        exc_info = {"message": "build_plan.compiler_selected is required for compiled languages."}
        term_outputs.append(exc_info["message"])
    elif available_compiler_names and selected_compiler not in available_compiler_names:
        exc_type = "CompilationError"
        exc_info = {
            "message": f"compiler_selected '{selected_compiler}' not in available_compilers.",
            "available_compilers": available_compiler_names,
        }
        term_outputs.append(exc_info["message"])

    # --- Compile phase (compiled languages only) ---
    if exc_type is None and not is_python_experiment:
        phases_executed.append("compile")
        if not compile_succeeds:
            exc_type = "CompilationError"
            exc_info = {"returncode": 1, "stderr": "compile error"}

    # --- Run phase (both Python and compiled languages) ---
    if exc_type is None:
        phases_executed.append("run")
        if not run_succeeds:
            exc_type = "RuntimeError"
            exc_info = {"returncode": 1, "stderr": "run error"}

    return exc_type, term_outputs, phases_executed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPythonExperimentRunPhase:
    """Verify Python experiments reach the run phase."""

    def test_python_experiment_skips_compile_but_runs(self):
        """Core regression: Python experiment must execute the run phase."""
        exc_type, outputs, phases = _simulate_compile_and_run(
            build_plan={"language": "python"},
            selected_compiler=None,
            available_compiler_names=None,
            compile_commands=[],
            run_commands=["python3 src/run_experiment.py"],
        )
        assert exc_type is None
        assert "run" in phases, "Python experiment must execute the run phase"
        assert "compile" not in phases, "Python experiment must skip compile"
        assert any("skipping compile phase" in o for o in outputs)

    def test_python_experiment_run_failure_propagates(self):
        """If run phase fails for Python experiment, exc_type must be set."""
        exc_type, _, phases = _simulate_compile_and_run(
            build_plan={"language": "python"},
            selected_compiler=None,
            available_compiler_names=None,
            compile_commands=[],
            run_commands=["python3 src/run_experiment.py"],
            run_succeeds=False,
        )
        assert exc_type == "RuntimeError"
        assert "run" in phases

    def test_compiled_language_runs_both_phases(self):
        """Compiled language should compile then run."""
        exc_type, _, phases = _simulate_compile_and_run(
            build_plan={"language": "c"},
            selected_compiler="gcc",
            available_compiler_names=["gcc", "icc"],
            compile_commands=["gcc -o main main.c"],
            run_commands=["./main"],
        )
        assert exc_type is None
        assert phases == ["compile", "run"]

    def test_compiled_language_compile_failure_skips_run(self):
        """If compile fails, run phase must not execute."""
        exc_type, _, phases = _simulate_compile_and_run(
            build_plan={"language": "c"},
            selected_compiler="gcc",
            available_compiler_names=["gcc"],
            compile_commands=["gcc -o main main.c"],
            run_commands=["./main"],
            compile_succeeds=False,
        )
        assert exc_type == "CompilationError"
        assert phases == ["compile"], "Run phase must not execute after compile failure"

    def test_missing_compiler_for_compiled_language(self):
        """Missing compiler must produce CompilationError without running."""
        exc_type, _, phases = _simulate_compile_and_run(
            build_plan={"language": "fortran"},
            selected_compiler=None,
            available_compiler_names=["gfortran"],
            compile_commands=["gfortran main.f90"],
            run_commands=["./a.out"],
        )
        assert exc_type == "CompilationError"
        assert phases == []

    def test_invalid_compiler_for_compiled_language(self):
        """Invalid compiler must produce CompilationError without running."""
        exc_type, _, phases = _simulate_compile_and_run(
            build_plan={"language": "c"},
            selected_compiler="clang",
            available_compiler_names=["gcc", "icc"],
            compile_commands=["clang -o main main.c"],
            run_commands=["./main"],
        )
        assert exc_type == "CompilationError"
        assert phases == []


class TestSourceCodeConsistency:
    """Verify the actual source matches the expected fix pattern."""

    def test_run_phase_not_nested_under_compiled_only_guard(self):
        """The run phase block must be at the same indent as the compile block,
        guarded only by `if exc_type is None:` (without `not is_python_experiment`)."""
        src = Path(__file__).resolve().parent.parent / "ai_scientist" / "treesearch" / "parallel_agent.py"
        if not src.exists():
            import pytest
            pytest.skip("Source file not found")
        content = src.read_text()

        # Find the run phase comment and its guard
        run_phase_match = re.search(
            r"^(\s+)# --- Run phase \(both Python and compiled languages\) ---\n"
            r"\1if exc_type is None:",
            content,
            re.MULTILINE,
        )
        assert run_phase_match is not None, (
            "Expected '# --- Run phase (both Python and compiled languages) ---' "
            "followed by 'if exc_type is None:' at the same indent level"
        )

        # The guard must NOT contain `not is_python_experiment`
        guard_line_start = run_phase_match.end()
        guard_line_end = content.index("\n", guard_line_start)
        guard_rest = content[guard_line_start:guard_line_end]
        assert "is_python_experiment" not in guard_rest, (
            "Run phase guard must not exclude Python experiments"
        )
