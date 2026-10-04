"""Command line: `agent-evals run`, `agent-evals compare`, `agent-evals list`."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from . import __version__
from .cases import CaseError, load_cases, load_world
from .compare import compare, render_comparison
from .judge import make_judge
from .models import MODEL_HELP, make_model_factory
from .report import build_report, load_report, render_summary, write_report
from .runner import RunResult, run_suite

DEFAULT_CASES = "evals/cases"
DEFAULT_WORLD = "evals/world.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals",
        description="Run eval cases against a tool-using agent and gate on regressions.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the eval suite and write a JSON report")
    run.add_argument("--model", default="scripted", help=f"model to test: {MODEL_HELP}")
    run.add_argument("--repeat", type=int, default=5, help="runs per case (default: 5)")
    run.add_argument("--seed", type=int, default=7, help="base seed for the scripted model")
    run.add_argument("--cases", default=DEFAULT_CASES, help="case file or directory")
    run.add_argument("--world", default=DEFAULT_WORLD, help="world fixture (YAML)")
    run.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    run.add_argument("--case", action="append", default=[], help="only case IDs matching this glob")
    run.add_argument(
        "--judge", default="none", help="LLM judge: none (default) or claude[:model-id]"
    )
    run.add_argument("--max-steps", type=int, default=10, help="step budget per run (default: 10)")
    run.add_argument(
        "--workers", type=int, default=1, help="parallel runs (useful for real models)"
    )
    run.add_argument("--out", default="reports/latest.json", help="where to write the JSON report")
    run.add_argument(
        "--min-pass-rate", type=float, help="exit 1 if the overall pass rate is below this (0-1)"
    )
    run.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v prints transcripts of failing runs, -vv of every run",
    )

    cmp = sub.add_parser("compare", help="compare a report with a baseline (the CI gate)")
    cmp.add_argument("baseline", help="baseline report, e.g. baselines/scripted.json")
    cmp.add_argument("current", nargs="?", default="reports/latest.json", help="report to check")
    cmp.add_argument(
        "--max-regression",
        type=float,
        default=0.03,
        help="allowed drop in pass rate, as a fraction (default: 0.03 = 3 points)",
    )

    lst = sub.add_parser("list", help="list the eval cases")
    lst.add_argument("--cases", default=DEFAULT_CASES)
    lst.add_argument("--tag", action="append", default=[])

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "compare":
            return _compare(args)
        return _list(args)
    except (CaseError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _select(cases, tags: list[str], patterns: list[str] | None = None):
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    if patterns:
        cases = [c for c in cases if any(fnmatch.fnmatch(c.id, p) for p in patterns)]
    if not cases:
        raise CaseError("no cases match the given --tag/--case filters")
    return cases


def _run(args: argparse.Namespace) -> int:
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    cases = _select(load_cases(args.cases), args.tag, args.case)
    world = load_world(args.world)
    factory = make_model_factory(args.model)
    judge = make_judge(args.judge)
    model_name = factory(0).name

    total = len(cases) * args.repeat
    print(f"Running {len(cases)} cases x {args.repeat} runs with {model_name}...", file=sys.stderr)
    progress = _Progress(total, enabled=sys.stderr.isatty() and not args.verbose)

    results = run_suite(
        cases,
        world,
        factory,
        repeat=args.repeat,
        seed=args.seed,
        judge=judge,
        max_steps=args.max_steps,
        workers=args.workers,
        on_result=progress.tick,
    )
    progress.done()

    if args.verbose:
        _print_transcripts(results, everything=args.verbose > 1)

    report = build_report(
        cases,
        results,
        meta={
            "model": model_name,
            "judge": judge.name if judge else None,
            "repeat": args.repeat,
            "seed": args.seed,
            "filters": {"tag": args.tag, "case": args.case},
        },
    )
    write_report(report, args.out)
    print(render_summary(report))
    print(f"\nReport written to {args.out}")

    if args.min_pass_rate is not None and report["summary"]["pass_rate"] < args.min_pass_rate:
        print(f"FAILED: pass rate below {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    return 0


def _compare(args: argparse.Namespace) -> int:
    result = compare(
        load_report(args.baseline), load_report(args.current), max_regression=args.max_regression
    )
    print(render_comparison(result))
    return 0 if result.ok else 1


def _list(args: argparse.Namespace) -> int:
    cases = _select(load_cases(args.cases), args.tag)
    width = max(len(c.id) for c in cases)
    for case in cases:
        tags = f"  [{', '.join(case.tags)}]" if case.tags else ""
        print(f"{case.id:{width}}  {case.input}{tags}")
    print(f"\n{len(cases)} cases from {Path(args.cases)}")
    return 0


def _print_transcripts(results: list[RunResult], everything: bool) -> None:
    for result in sorted(results, key=lambda r: (r.case_id, r.run)):
        if result.passed and not everything:
            continue
        status = "PASS" if result.passed else "FAIL"
        print(f"--- {result.case_id} run {result.run}: {status}")
        if result.transcript:
            print(result.transcript.render())
        for check in result.failures:
            print(f"  ✗ {check.describe()}")
        print()


class _Progress:
    def __init__(self, total: int, enabled: bool):
        self.total, self.count, self.enabled = total, 0, enabled

    def tick(self, _result: RunResult) -> None:
        self.count += 1
        if self.enabled:
            print(f"\r  {self.count}/{self.total} runs", end="", file=sys.stderr, flush=True)

    def done(self) -> None:
        if self.enabled:
            print("\r" + " " * 30 + "\r", end="", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
