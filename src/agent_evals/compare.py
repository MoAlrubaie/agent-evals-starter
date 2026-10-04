"""The regression gate: compare a report against a baseline.

Gate on regressions, not perfection. The build fails when the pass rate over
the cases both reports share drops by more than the tolerance, and the output
lists exactly which cases changed so the failure is a conversation, not a
mystery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Comparison:
    ok: bool
    baseline_rate: float
    current_rate: float
    max_regression: float
    regressed: list[tuple[str, float, float]] = field(default_factory=list)
    improved: list[tuple[str, float, float]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.current_rate - self.baseline_rate


def compare(
    baseline: dict[str, Any], current: dict[str, Any], *, max_regression: float = 0.03
) -> Comparison:
    base_cases, cur_cases = baseline["cases"], current["cases"]
    shared = [case_id for case_id in cur_cases if case_id in base_cases]

    def rate(cases: dict[str, Any]) -> float:
        runs = sum(cases[c]["runs"] for c in shared)
        passed = sum(cases[c]["passed"] for c in shared)
        return passed / runs if runs else 0.0

    result = Comparison(
        ok=True,
        baseline_rate=rate(base_cases),
        current_rate=rate(cur_cases),
        max_regression=max_regression,
        added=[c for c in cur_cases if c not in base_cases],
        removed=[c for c in base_cases if c not in cur_cases],
    )
    for case_id in shared:
        before, after = base_cases[case_id]["pass_rate"], cur_cases[case_id]["pass_rate"]
        if after < before:
            result.regressed.append((case_id, before, after))
        elif after > before:
            result.improved.append((case_id, before, after))

    if baseline.get("model") != current.get("model"):
        result.warnings.append(
            f"comparing different models: baseline={baseline.get('model')} "
            f"current={current.get('model')}"
        )
    if baseline.get("repeat") != current.get("repeat"):
        result.warnings.append(
            f"different repeat counts: baseline={baseline.get('repeat')} "
            f"current={current.get('repeat')}"
        )
    if not shared:
        result.warnings.append("no cases in common; nothing to compare")

    result.ok = bool(shared) and result.delta >= -max_regression - 1e-9
    return result


def render_comparison(result: Comparison) -> str:
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    lines = [f"warning: {w}" for w in result.warnings]
    lines.append(
        f"Pass rate {pct(result.baseline_rate)} -> {pct(result.current_rate)} "
        f"({'+' if result.delta >= 0 else ''}{result.delta * 100:.1f} pts, "
        f"tolerance -{result.max_regression * 100:.1f} pts)"
    )
    if result.regressed:
        lines.append("Regressed:")
        lines += [f"  ✗ {c:34} {pct(b):>7} -> {pct(a)}" for c, b, a in result.regressed]
    if result.improved:
        lines.append("Improved:")
        lines += [f"  ✓ {c:34} {pct(b):>7} -> {pct(a)}" for c, b, a in result.improved]
    if result.added:
        lines.append(
            f"New cases (not gated until the baseline is updated): {', '.join(result.added)}"
        )
    if result.removed:
        lines.append(f"Cases missing from this run: {', '.join(result.removed)}")
    if not (result.regressed or result.improved):
        lines.append("No case changed its pass rate.")
    lines.append(
        "GATE PASSED" if result.ok else "GATE FAILED: pass rate regressed beyond tolerance"
    )
    return "\n".join(lines)
