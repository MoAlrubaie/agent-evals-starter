"""Run every case N times against a fresh world and collect the results."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .agent import Transcript, run_agent
from .cases import Case
from .checks import Check, evaluate
from .models.base import Model
from .world import World


@dataclass
class RunResult:
    case_id: str
    run: int
    passed: bool
    checks: list[Check] = field(default_factory=list)
    transcript: Transcript | None = None
    latency_ms: float = 0.0
    error: str = ""

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.skipped]


def run_seed(seed: int, case_id: str, run: int) -> int:
    """A stable per-run seed (Python's hash() is randomized per process, so don't use it)."""
    digest = hashlib.sha256(f"{seed}:{case_id}:{run}".encode()).hexdigest()
    return int(digest[:8], 16)


def run_case_once(
    case: Case,
    run: int,
    world_base: dict[str, Any],
    model_factory: Callable[[int], Model],
    *,
    seed: int = 0,
    judge=None,
    max_steps: int = 10,
) -> RunResult:
    world = World.build(world_base, case.setup)
    model = model_factory(run_seed(seed, case.id, run))
    started = time.perf_counter()
    try:
        transcript = run_agent(model, world, case.input, max_steps=max_steps)
    except Exception as exc:  # a crashed run is a failed run, not a crashed suite
        elapsed = (time.perf_counter() - started) * 1000
        error = f"{type(exc).__name__}: {exc}"
        check = Check("trajectory", "run completed without an error", False, error)
        return RunResult(case.id, run, False, [check], None, elapsed, error)
    elapsed = (time.perf_counter() - started) * 1000

    checks = evaluate(case, world.snapshot(), transcript, judge)
    passed = all(c.passed for c in checks if not c.skipped)
    return RunResult(case.id, run, passed, checks, transcript, elapsed)


def run_suite(
    cases: list[Case],
    world_base: dict[str, Any],
    model_factory: Callable[[int], Model],
    *,
    repeat: int = 5,
    seed: int = 0,
    judge=None,
    max_steps: int = 10,
    workers: int = 1,
    on_result: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    jobs = [(case, run) for case in cases for run in range(repeat)]

    def work(job: tuple[Case, int]) -> RunResult:
        case, run = job
        result = run_case_once(
            case, run, world_base, model_factory, seed=seed, judge=judge, max_steps=max_steps
        )
        if on_result:
            on_result(result)
        return result

    if workers <= 1:
        return [work(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, jobs))
