"""Loading eval cases from YAML.

A case is data: an input, an optional patch to the world, and what "correct"
means, written down before anyone looks at the agent's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXPECT_KEYS = {
    "tools_called": "tools that must be called at least once",
    "tools_not_called": "tools that must never be called",
    "tool_order": "tools that must be called in this relative order",
    "tool_args": "per tool, arguments at least one call must have",
    "final_state": "dotted path -> expected value in the world afterwards",
    "escalated": "whether the agent handed off to a human",
    "reply_contains": "phrases the final reply must contain (case-insensitive)",
    "reply_not_contains": "phrases the final reply must not contain (case-insensitive)",
    "max_steps": "the most model turns the case may take",
}


class CaseError(ValueError):
    pass


@dataclass
class Case:
    id: str
    input: str
    expect: dict[str, Any] = field(default_factory=dict)
    setup: dict[str, Any] = field(default_factory=dict)
    judge: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    source: str = ""


def load_world(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise CaseError(f"{path}: the world file must be a mapping")
    return data


def load_cases(path: str | Path) -> list[Case]:
    """Load every case from a YAML file, or from all *.yaml files in a directory."""
    path = Path(path)
    files = sorted([*path.glob("*.yaml"), *path.glob("*.yml")]) if path.is_dir() else [path]
    if not files:
        raise CaseError(f"no case files found in {path}")

    cases: list[Case] = []
    seen: dict[str, str] = {}
    for file in files:
        raw = yaml.safe_load(file.read_text()) or []
        if not isinstance(raw, list):
            raise CaseError(f"{file}: expected a list of cases")
        for index, item in enumerate(raw):
            case = _parse_case(item, f"{file}[{index}]")
            if case.id in seen:
                raise CaseError(f"{file}: duplicate case id {case.id!r} (also in {seen[case.id]})")
            seen[case.id] = str(file)
            cases.append(case)
    return cases


def _parse_case(item: Any, where: str) -> Case:
    if not isinstance(item, dict):
        raise CaseError(f"{where}: each case must be a mapping")
    for required in ("id", "input"):
        if not item.get(required):
            raise CaseError(f"{where}: missing {required!r}")
    where = f"{where} ({item['id']})"

    known = {"id", "input", "expect", "setup", "judge", "tags", "description"}
    unknown = set(item) - known
    if unknown:
        raise CaseError(f"{where}: unknown field(s) {sorted(unknown)}")

    expect = item.get("expect") or {}
    bad = set(expect) - set(EXPECT_KEYS)
    if bad:
        raise CaseError(
            f"{where}: unknown expect key(s) {sorted(bad)}; known: {sorted(EXPECT_KEYS)}"
        )
    if not expect and not item.get("judge"):
        raise CaseError(f"{where}: a case needs at least one expectation or judge question")

    judge = item.get("judge") or []
    if isinstance(judge, str):
        judge = [judge]
    return Case(
        id=str(item["id"]),
        input=str(item["input"]),
        expect=expect,
        setup=item.get("setup") or {},
        judge=[str(q) for q in judge],
        tags=[str(t) for t in item.get("tags") or []],
        description=str(item.get("description") or ""),
        source=where,
    )
