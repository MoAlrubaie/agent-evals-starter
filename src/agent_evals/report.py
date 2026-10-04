"""Turn run results into a JSON report and a readable console summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import Case
from .runner import RunResult

SCHEMA_VERSION = 1
MAX_FAILURES_KEPT = 3


def build_report(
    cases: list[Case], results: list[RunResult], *, meta: dict[str, Any]
) -> dict[str, Any]:
    by_case: dict[str, list[RunResult]] = {case.id: [] for case in cases}
    for result in results:
        by_case[result.case_id].append(result)

    case_reports: dict[str, Any] = {}
    for case in cases:
        runs = sorted(by_case[case.id], key=lambda r: r.run)
        passed = sum(r.passed for r in runs)
        steps = [r.transcript.steps for r in runs if r.transcript]
        failures = [
            {"run": r.run, "checks": [c.describe() for c in r.failures]}
            for r in runs
            if not r.passed
        ]
        case_reports[case.id] = {
            "tags": case.tags,
            "runs": len(runs),
            "passed": passed,
            "pass_rate": _ratio(passed, len(runs)),
            "all_passed": passed == len(runs),
            "mean_steps": round(sum(steps) / len(steps), 2) if steps else None,
            "tokens": sum(r.transcript.usage.total for r in runs if r.transcript),
            "judge_skipped": sum(1 for r in runs for c in r.checks if c.skipped),
            "failures": failures[:MAX_FAILURES_KEPT],
        }

    total_runs = len(results)
    passed_runs = sum(r.passed for r in results)
    all_steps = [r.transcript.steps for r in results if r.transcript]
    latencies = sorted(r.latency_ms for r in results)
    summary = {
        "cases": len(cases),
        "runs": total_runs,
        "passed_runs": passed_runs,
        "pass_rate": _ratio(passed_runs, total_runs),
        "cases_all_passed": sum(c["all_passed"] for c in case_reports.values()),
        "pass_all_rate": _ratio(sum(c["all_passed"] for c in case_reports.values()), len(cases)),
        "mean_steps": round(sum(all_steps) / len(all_steps), 2) if all_steps else None,
        "tokens": sum(c["tokens"] for c in case_reports.values()),
        "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else None,
        "judge_skipped": sum(c["judge_skipped"] for c in case_reports.values()),
    }
    return {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **meta,
        "summary": summary,
        "cases": case_reports,
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def load_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    if report.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported report schema {report.get('schema')!r}")
    return report


def render_summary(report: dict[str, Any]) -> str:
    s = report["summary"]
    repeat = report.get("repeat", "?")
    lines = [
        f"agent-evals  model={report.get('model')}  cases={s['cases']}  "
        f"repeat={repeat}  seed={report.get('seed')}",
        "",
        f"  {'':2}{'case':34}{'passed':>8}{'rate':>7}{'steps':>7}",
    ]
    for case_id, c in report["cases"].items():
        mark = "✓" if c["all_passed"] else ("✗" if c["passed"] == 0 else "~")
        steps = f"{c['mean_steps']:.1f}" if c["mean_steps"] is not None else "-"
        passed = f"{c['passed']:>5}/{c['runs']:<2}"
        lines.append(f"  {mark} {case_id:34}{passed}{_pct(c['pass_rate']):>7}{steps:>7}")

    failing = {k: v for k, v in report["cases"].items() if v["failures"]}
    if failing:
        lines += ["", "Failures (first failing run of each case):"]
        for case_id, c in failing.items():
            first = c["failures"][0]
            lines.append(f"  {case_id} (run {first['run']}):")
            lines += [f"    - {check}" for check in first["checks"]]

    lines += [
        "",
        f"Pass rate {_pct(s['pass_rate'])} ({s['passed_runs']}/{s['runs']} runs)   "
        f"all {repeat} runs passed: {s['cases_all_passed']}/{s['cases']} cases   "
        f"mean steps {s['mean_steps']}   tokens {s['tokens']:,}",
    ]
    if s["judge_skipped"]:
        lines.append(
            f"{s['judge_skipped']} judge checks skipped (no judge configured; see --judge)."
        )
    return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"
