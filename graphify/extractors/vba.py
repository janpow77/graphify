"""Conservative structural extraction for exported VBA standard modules (.bas)."""

from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.base import _file_stem, _make_id

_MODULE = re.compile(r'^\s*Attribute\s+VB_Name\s*=\s*"([^"]+)"', re.IGNORECASE)
_PROCEDURE = re.compile(
    r"^\s*(?:(?:Public|Private|Friend|Static)\s+)?"
    r"(Sub|Function|Property\s+(?:Get|Let|Set))\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.IGNORECASE,
)
_END = re.compile(r"^\s*End\s+(?:Sub|Function|Property)\b", re.IGNORECASE)
_CALL = re.compile(r"^\s*Call\s+([A-Za-z_][A-Za-z_0-9]*)\b", re.IGNORECASE)
_BARE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\b(.*)$")


def _without_comment(line: str) -> str:
    """Remove VBA apostrophe comments without cutting quoted strings."""
    quoted = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == '"':
            if quoted and index + 1 < len(line) and line[index + 1] == '"':
                index += 2
                continue
            quoted = not quoted
        elif char == "'" and not quoted:
            return line[:index]
        index += 1
    return line


def extract_vba(path: Path) -> dict:
    """Extract VBA module/procedure nodes and local procedure calls."""
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}
    source_file = str(path)
    stem = _file_stem(path)
    module_name = path.stem
    for line in lines:
        match = _MODULE.match(line)
        if match:
            module_name = match.group(1)
            break

    nodes: list[dict] = []
    edges: list[dict] = []
    file_id = _make_id(source_file)
    module_id = _make_id(stem, "module", module_name)

    def node(identifier: str, label: str, line: int, kind: str) -> None:
        nodes.append(
            {
                "id": identifier,
                "label": label,
                "file_type": "code",
                "source_file": source_file,
                "source_location": f"L{line}",
                "metadata": {"language": "vba", "kind": kind},
            }
        )

    def edge(source: str, target: str, relation: str, line: int) -> None:
        edges.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": "EXTRACTED",
                "source_file": source_file,
                "source_location": f"L{line}",
                "weight": 1.0,
            }
        )

    node(file_id, path.name, 1, "file")
    node(module_id, module_name, 1, "module")
    edge(file_id, module_id, "contains", 1)
    definitions: dict[str, str] = {}
    starts: dict[int, str] = {}
    for line_number, raw in enumerate(lines, 1):
        match = _PROCEDURE.match(_without_comment(raw))
        if not match or _MODULE.match(raw):
            continue
        kind = match.group(1).casefold().replace(" ", "_")
        name = match.group(2)
        key = name.casefold()
        identifier = _make_id(stem, kind if kind.startswith("property") else "procedure", key)
        if identifier in definitions.values():
            continue
        definitions[key] = identifier
        starts[line_number] = identifier
        node(identifier, f"{name}()", line_number, kind)
        edge(module_id, identifier, "contains", line_number)

    active: str | None = None
    seen_calls: set[tuple[str, str, int]] = set()
    for line_number, raw in enumerate(lines, 1):
        if line_number in starts:
            active = starts[line_number]
            continue
        if active is None:
            continue
        line = _without_comment(raw).strip()
        if _END.match(line):
            active = None
            continue
        match = _CALL.match(line)
        if match is None:
            bare = _BARE.match(line)
            if bare is None or bare.group(2).lstrip().startswith("="):
                continue
            name = bare.group(1)
        else:
            name = match.group(1)
        target = definitions.get(name.casefold())
        key = (active, target or "", line_number)
        if target and target != active and key not in seen_calls:
            seen_calls.add(key)
            edge(active, target, "calls", line_number)
    return {"nodes": nodes, "edges": edges}
