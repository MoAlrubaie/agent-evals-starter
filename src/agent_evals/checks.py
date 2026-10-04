"""Checks: turn a finished run into a list of pass/fail results.

Most checks are deterministic and look at outcomes (the state of the world)
and the trajectory (which tools ran, in what order, how many steps). Judge
checks are the thin top layer and only run when a judge is configured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .agent import Transcript
from .cases import Case

# Layers of the eval pyramid, cheapest first.
LAYERS = ("state", "trajectory", "reply", "judge")


@dataclass
class Check:
    layer: str
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False

    def describe(self) -> str:
        text = f"{self.layer}: {self.name}"
        return f"{text} ({self.detail})" if self.detail else text


def evaluate(case: Case, world: dict[str, Any], transcript: Transcript, judge=None) -> list[Check]:
    checks = _builtin_checks(transcript)
    expect = case.expect
    called = [call.name for call in transcript.tool_calls]

    for tool in expect.get("tools_called", []):
        checks.append(Check("trajectory", f"calls {tool}", tool in called, _called_detail(called)))

    for tool in expect.get("tools_not_called", []):
        count = called.count(tool)
        checks.append(
            Check(
                "trajectory", f"never calls {tool}", count == 0, f"called {count}x" if count else ""
            )
        )

    if "tool_order" in expect:
        order = list(expect["tool_order"])
        checks.append(
            Check(
                "trajectory",
                f"calls {' -> '.join(order)} in order",
                _is_subsequence(order, called),
                _called_detail(called),
            )
        )

    for tool, wanted in (expect.get("tool_args") or {}).items():
        calls = [c.args for c in transcript.tool_calls if c.name == tool]
        ok = any(all(_same(args.get(k), v) for k, v in wanted.items()) for args in calls)
        detail = "" if ok else (f"got {calls}" if calls else f"{tool} was not called")
        checks.append(Check("trajectory", f"{tool} args match {wanted}", ok, detail))

    if "max_steps" in expect:
        limit = int(expect["max_steps"])
        checks.append(
            Check(
                "trajectory",
                f"at most {limit} steps",
                transcript.steps <= limit,
                f"took {transcript.steps}",
            )
        )

    for path, wanted in (expect.get("final_state") or {}).items():
        found, actual = get_path(world, str(path))
        ok = found and _same(actual, wanted)
        detail = "" if ok else (f"was {actual!r}" if found else "path not found")
        checks.append(Check("state", f"{path} == {wanted!r}", ok, detail))

    if "escalated" in expect:
        escalated = bool(world.get("escalations"))
        wanted = bool(expect["escalated"])
        name = "escalated to a human" if wanted else "did not escalate"
        checks.append(Check("state", name, escalated == wanted))

    reply = transcript.reply.lower()
    for phrase in expect.get("reply_contains", []):
        ok = str(phrase).lower() in reply
        checks.append(
            Check("reply", f"mentions {phrase!r}", ok, "" if ok else _quote(transcript.reply))
        )
    for phrase in expect.get("reply_not_contains", []):
        ok = str(phrase).lower() not in reply
        checks.append(
            Check("reply", f"doesn't say {phrase!r}", ok, "" if ok else _quote(transcript.reply))
        )

    for question in case.judge:
        if judge is None:
            checks.append(Check("judge", question, True, "no judge configured", skipped=True))
            continue
        verdict, reason = judge.ask(question, transcript)
        checks.append(Check("judge", question, verdict, "" if verdict else reason))

    return checks


def _builtin_checks(transcript: Transcript) -> list[Check]:
    """Checks every case gets: the base of the pyramid."""
    invalid = [f"{c.name}: {c.invalid}" for c in transcript.tool_calls if c.invalid]
    checks = [
        Check("trajectory", "tool calls match their schemas", not invalid, "; ".join(invalid)),
        Check(
            "trajectory",
            "finished within the step budget",
            transcript.stop != "max_steps",
            f"stopped after {transcript.steps} steps" if transcript.stop == "max_steps" else "",
        ),
    ]
    if transcript.stop == "refusal":
        checks.append(Check("reply", "model did not refuse", False, "stop_reason=refusal"))
    else:
        checks.append(Check("reply", "gave a final reply", bool(transcript.reply)))
    return checks


def get_path(data: Any, path: str) -> tuple[bool, Any]:
    """Resolve "orders.1004.refunded" against nested dicts (and list indexes)."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _same(actual: Any, expected: Any) -> bool:
    numeric = (int, float)
    if (
        isinstance(actual, numeric)
        and isinstance(expected, numeric)
        and not isinstance(actual, bool)
    ):
        return math.isclose(float(actual), float(expected), abs_tol=0.005)
    if isinstance(expected, str) and not isinstance(actual, str) and actual is not None:
        return str(actual) == expected
    return actual == expected


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    remaining = iter(haystack)
    return all(item in remaining for item in needle)


def _called_detail(called: list[str]) -> str:
    return f"called: {', '.join(called) or 'nothing'}"


def _quote(text: str, limit: int = 100) -> str:
    text = text.replace("\n", " ")
    return f'reply: "{text[:limit]}{"..." if len(text) > limit else ""}"'
