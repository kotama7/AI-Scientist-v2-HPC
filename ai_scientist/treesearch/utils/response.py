import json
import re

import black


def wrap_code(code: str, lang: str | None = "python") -> str:
    """Wrap code with triple backticks and an optional language hint."""
    lang = lang or ""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script: str) -> bool:
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text: str) -> list[dict]:
    """Extract all JSON objects from the text. Caveat: cannot handle nested JSON."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    """Trim a long string, keeping the first/last k characters with ellipsis."""
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    return string


def extract_code(text: str, language: str = "python") -> str:
    """Extract code blocks from the text, respecting the desired language."""
    code_blocks = []
    pattern = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)
    for match in pattern.findall(text):
        code_blocks.append(match.strip())

    if not code_blocks:
        fallback = re.findall(r"^(```\w+)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if fallback:
            code_blocks.append(fallback[0][1].strip())

    if language.lower() == "python":
        valid_blocks = [format_code(c) for c in code_blocks if is_valid_python_script(c)]
        return format_code("\n\n".join(valid_blocks))

    return "\n\n".join(code_blocks).strip()


def extract_text_up_to_code(s: str) -> str:
    """Extract natural language text up to the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code: str) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
