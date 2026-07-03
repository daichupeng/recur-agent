"""Skill library: read/write/search SKILL.md files in the skill_lib directory."""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SkillEntry:
    """One skill loaded from skill_lib/<name>/SKILL.md."""
    name: str                   # frontmatter `name:` slug
    description: str            # frontmatter `description:` text
    exec_type: str              # one of DETERMINISTIC_CODE / EXTERNAL_API / OPENSOURCE_LIBRARY / LLM_PROMPT
    implementation: str         # fenced Python block (tool skills only)
    instruction: str            # fenced instruction block (LLM skills only)
    input_schema: Optional[dict] = None   # JSON Schema dict extracted from body
    output_schema: Optional[dict] = None  # JSON Schema dict extracted from body
    skill_dir: Optional[Path] = None      # path to skill_lib/<name>/


def _extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML frontmatter from the Markdown body.

    Returns (frontmatter_dict, body). Only string scalars and multi-line
    `description: >` are supported — enough for our needs without a full YAML dep.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    front_raw = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")

    result: dict[str, str] = {}
    current_key: str | None = None
    fold_lines: list[str] = []

    for line in front_raw.splitlines():
        # Multi-line folded value continuation (starts with spaces)
        if current_key and line.startswith("  "):
            fold_lines.append(line.strip())
            continue
        if fold_lines:
            result[current_key] = " ".join(fold_lines)  # type: ignore[index]
            fold_lines = []
            current_key = None

        m = re.match(r"^(\w+):\s*(>?)(.*)", line)
        if not m:
            continue
        key, fold_marker, rest = m.group(1), m.group(2), m.group(3).strip()
        if fold_marker == ">":
            current_key = key
            fold_lines = [rest] if rest else []
        else:
            result[key] = rest

    if fold_lines and current_key:
        result[current_key] = " ".join(fold_lines)

    return result, body


def _extract_fenced_block(body: str, lang: str) -> str:
    """Return the contents of the first ```<lang>…``` fenced block, or empty string."""
    pattern = re.compile(
        r"```" + re.escape(lang) + r"\n(.*?)```",
        re.DOTALL,
    )
    m = pattern.search(body)
    return m.group(1).rstrip() if m else ""


def _extract_json_schema(body: str, label: str) -> Optional[dict]:
    """Extract JSON Schema from a labeled ```json block (e.g. ## Input Schema)."""
    import json
    pattern = re.compile(
        r"##\s+" + re.escape(label) + r".*?\n```json\n(.*?)```",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(body)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _similarity(a: str, b: str) -> float:
    """Very lightweight word-overlap similarity in [0, 1]."""
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class SkillLib:
    """Read/write/search the skill_lib directory.

    Each skill lives in skill_lib/<name>/SKILL.md.
    The Python implementation is embedded in a ```python fenced block.
    """

    SIMILARITY_THRESHOLD = 0.35

    def __init__(self, lib_dir: Path) -> None:
        self._lib_dir = lib_dir
        lib_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, SkillEntry] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, name: str, description: str, top_k: int = 3) -> list[SkillEntry]:
        """Return the top-k skills whose description best matches the query.

        Only entries above SIMILARITY_THRESHOLD are returned.
        """
        query = f"{name} {description}"
        scored = [
            (entry, _similarity(query, f"{entry.name} {entry.description}"))
            for entry in self._cache.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, score in scored[:top_k] if score >= self.SIMILARITY_THRESHOLD]

    def get(self, name: str) -> Optional[SkillEntry]:
        return self._cache.get(name)

    def save(self, entry: SkillEntry) -> Path:
        """Persist a SkillEntry to skill_lib/<name>/SKILL.md and return the path."""
        import json
        skill_dir = self._lib_dir / entry.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"

        desc_lines = textwrap.fill(entry.description, width=78)
        desc_indented = "\n  ".join(desc_lines.splitlines())

        lines: list[str] = [
            "---",
            f"name: {entry.name}",
            "description: >",
            f"  {desc_indented}",
            f"exec_type: {entry.exec_type}",
            "---",
            "",
        ]

        if entry.input_schema:
            lines += [
                "## Input Schema",
                "",
                "```json",
                json.dumps(entry.input_schema, indent=2),
                "```",
                "",
            ]

        if entry.output_schema:
            lines += [
                "## Output Schema",
                "",
                "```json",
                json.dumps(entry.output_schema, indent=2),
                "```",
                "",
            ]

        if entry.exec_type == "LLM_PROMPT" and entry.instruction:
            lines += [
                "## Instruction",
                "",
                "```instruction",
                entry.instruction,
                "```",
                "",
            ]
        elif entry.implementation:
            lines += [
                "## Implementation",
                "",
                "```python",
                entry.implementation,
                "```",
                "",
            ]

        path.write_text("\n".join(lines))
        entry.skill_dir = skill_dir
        self._cache[entry.name] = entry
        return path

    def all_skills(self) -> list[SkillEntry]:
        return list(self._cache.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        for skill_md in self._lib_dir.glob("*/SKILL.md"):
            entry = _parse_skill_md(skill_md)
            if entry:
                self._cache[entry.name] = entry

    def reload(self) -> None:
        self._cache.clear()
        self._load_all()


def _parse_skill_md(path: Path) -> Optional[SkillEntry]:
    """Parse a single SKILL.md file into a SkillEntry. Returns None on parse error."""
    try:
        text = path.read_text()
        front, body = _extract_frontmatter(text)

        name = front.get("name", "")
        description = front.get("description", "")
        exec_type = front.get("exec_type", "LLM_PROMPT")

        if not name:
            return None

        implementation = _extract_fenced_block(body, "python")
        instruction = _extract_fenced_block(body, "instruction")
        input_schema = _extract_json_schema(body, "Input Schema")
        output_schema = _extract_json_schema(body, "Output Schema")

        return SkillEntry(
            name=name,
            description=description,
            exec_type=exec_type,
            implementation=implementation,
            instruction=instruction,
            input_schema=input_schema,
            output_schema=output_schema,
            skill_dir=path.parent,
        )
    except Exception:
        return None
