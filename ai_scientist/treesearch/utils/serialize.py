import copy
import json
from pathlib import Path
from typing import Type, TypeVar
import re

import dataclasses_json

from ..journal import Journal, Node


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {}
        for n in obj.nodes:
            if n.parent is not None:
                # Handle both Node objects and string IDs
                parent_id = n.parent.id if isinstance(n.parent, Node) else n.parent
                node2parent[n.id] = parent_id
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to AIDE dataclasses."""
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)


def parse_markdown_to_dict(content: str):
    """
    Parse markdown-like text into a dictionary.

    Supported formats:
      - Lines like: "Key": "Value",
      - Markdown headings with "## Section" and optional "### Subsection",
        where list items are prefixed with "- ".
    """

    if content is None:
        return {}

    pattern = r'"([^"]+)"\s*:\s*"([^"]*?)"(?:,\s*|\s*$)'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    if matches:
        data_dict = {}
        for key, value in matches:
            data_dict[key] = value
        return data_dict

    header_pattern = re.compile(r"^\s*##\s+(.*)$")
    subheader_pattern = re.compile(r"^\s*###\s+(.*)$")

    def strip_blank_edges(lines: list[str]) -> list[str]:
        start = 0
        end = len(lines)
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        return lines[start:end]

    def lines_to_value(lines: list[str]):
        lines = strip_blank_edges(lines)
        if not lines:
            return ""
        non_empty = [line for line in lines if line.strip()]
        bullet_matches = [re.match(r"^\s*-\s+(.*)$", line) for line in non_empty]
        if bullet_matches and all(bullet_matches):
            return [match.group(1).strip() for match in bullet_matches]
        return "\n".join(lines).strip()

    def parse_section(lines: list[str]):
        lines = strip_blank_edges(lines)
        if not lines:
            return ""
        subsections = []
        current_key = None
        buffer: list[str] = []
        for line in lines:
            match = subheader_pattern.match(line)
            if match:
                if current_key is not None:
                    subsections.append((current_key, buffer))
                current_key = match.group(1).strip()
                buffer = []
            else:
                buffer.append(line)
        if current_key is not None:
            subsections.append((current_key, buffer))
        if not subsections:
            return lines_to_value(lines)
        sub_dict = {}
        for key, sub_lines in subsections:
            sub_dict[key] = lines_to_value(sub_lines)
        return sub_dict

    sections = []
    current_key = None
    buffer: list[str] = []
    for line in content.splitlines():
        match = header_pattern.match(line)
        if match:
            if current_key is not None:
                sections.append((current_key, buffer))
            current_key = match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)
    if current_key is not None:
        sections.append((current_key, buffer))

    if not sections:
        return {}

    data_dict = {}
    for key, section_lines in sections:
        data_dict[key] = parse_section(section_lines)
    return data_dict
