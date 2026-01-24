import json
import re

import black


class MalformedMemoryUpdateError(Exception):
    """Raised when a malformed <memory_update> block is detected."""
    pass


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


def extract_memory_updates(text: str) -> dict | None:
    """Extract memory update instructions from LLM response.

    Looks for a <memory_update>...</memory_update> block containing JSON
    that specifies memory operations for core, archival, and/or recall.

    Args:
        text: The LLM response text to parse.

    Returns:
        A dict with memory update instructions, or None if not found or invalid.
        Expected format:
        {
            "core": {"key": "value", ...},
            "archival": [{"text": "...", "tags": [...]}],
            "recall": {"kind": "...", "content": "..."}
        }
    """
    if not text:
        return None

    pattern = r'<memory_update>\s*(.*?)\s*</memory_update>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    try:
        updates = json.loads(match.group(1))
        if not isinstance(updates, dict):
            return None
        return updates
    except json.JSONDecodeError:
        return None


def check_malformed_memory_update(text: str) -> bool:
    """Check if text contains a malformed <memory_update> block.

    Malformed patterns include:
    - Missing closing '>' on opening tag: <memory_update"core":...
    - Self-closing with }}/> instead of proper closing tag

    Note: Escaped slash in closing tag (<\/memory_update>) is NOT malformed,
    it's a valid format handled by remove_memory_update_tags.

    Args:
        text: The text to check.

    Returns:
        True if malformed memory_update block detected, False otherwise.
    """
    if not text:
        return False
    # Pattern for malformed: <memory_update without proper > before content, ending with }}/>
    # This catches cases like: <memory_update"core":{...}}/>
    malformed_pattern = r'<memory_update[^>]*\{.*?\}\}\s*/>'
    return bool(re.search(malformed_pattern, text, flags=re.DOTALL))


def remove_memory_update_tags(text: str) -> str:
    """Remove <memory_update>...</memory_update> blocks from text.

    Handles various malformed patterns from LLM output:
    - Normal: <memory_update>...</memory_update>
    - Escaped slash: <memory_update>...<\/memory_update>
    - Malformed: <memory_update...}}/>

    Args:
        text: The text to clean.

    Returns:
        Text with memory update blocks removed.
    """
    if not text:
        return text
    # Match <memory_update with various endings:
    # - </memory_update> (normal)
    # - <\/memory_update> (escaped slash)
    # - }}/> (malformed self-closing)
    pattern = r'<memory_update.*?(?:</memory_update>|<\\/memory_update>|\}\}\s*/>)'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()
