"""LaTeX utilities for the output module."""

import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import uuid


def extract_latex_snippet(text: str) -> str:
    """Extract LaTeX code from text.

    Looks for LaTeX code in fenced blocks or raw content.

    Args:
        text: Text that may contain LaTeX code.

    Returns:
        Extracted LaTeX code string.
    """
    if not text:
        return ""

    fenced_latex = re.findall(
        r"```[ \t]*latex[ \t]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for match in fenced_latex:
        if match.strip():
            return match.strip()

    fenced_any = re.findall(
        r"```[ \t]*[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    for match in fenced_any:
        if match.strip():
            return match.strip()

    open_fence = re.search(
        r"```[ \t]*latex[ \t]*\r?\n", text, flags=re.IGNORECASE
    )
    if not open_fence:
        open_fence = re.search(r"```[ \t]*[a-zA-Z0-9_-]*[ \t]*\r?\n", text)
    if open_fence:
        remainder = text[open_fence.end() :].strip()
        if remainder:
            return remainder

    if "\\documentclass" in text or "\\begin{document}" in text:
        return text.strip()

    return ""


def compile_latex(cwd: str, pdf_file: str, timeout: int = 30) -> None:
    """Compile LaTeX document to PDF.

    Runs pdflatex and bibtex commands to generate PDF.

    Args:
        cwd: Working directory containing LaTeX files.
        pdf_file: Output PDF file path.
        timeout: Timeout in seconds for each command.
    """
    print("GENERATING LATEX")

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(
                f"EXCEPTION in compile_latex: LaTeX timed out after {timeout} seconds."
            )
            print(traceback.format_exc())
        except subprocess.CalledProcessError:
            print(
                f"EXCEPTION in compile_latex: Error running command {' '.join(command)}"
            )
            print(traceback.format_exc())

    print("FINISHED GENERATING LATEX")

    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")
        print("EXCEPTION in compile_latex while moving PDF:")
        print(traceback.format_exc())


def detect_pages_before_impact(latex_folder: str, timeout: int = 30) -> tuple[int, int] | None:
    """Detect the page where 'Impact Statement' appears.

    Temporarily compiles the LaTeX document and searches for the
    Impact Statement section.

    Args:
        latex_folder: Directory containing LaTeX source files.
        timeout: Timeout in seconds for compilation.

    Returns:
        Tuple of (page_number, line_number) if found, None otherwise.
    """
    temp_dir = osp.join(latex_folder, f"_temp_compile_{uuid.uuid4().hex}")
    try:
        shutil.copytree(latex_folder, temp_dir, dirs_exist_ok=True)

        # Compile in the temp folder
        commands = [
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["bibtex", "template"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ]
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return None

        temp_pdf_file = osp.join(temp_dir, "template.pdf")
        if not osp.exists(temp_pdf_file):
            return None

        # Try page-by-page extraction to detect "Impact Statement"
        for i in range(1, 51):
            page_txt = osp.join(temp_dir, f"page_{i}.txt")
            subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(i),
                    "-l",
                    str(i),
                    "-q",
                    temp_pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                break
            with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                page_content = fp.read()
            lines = page_content.split("\n")
            for idx, line in enumerate(lines):
                if "Impact Statement" in line:
                    return (i, idx + 1)
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
