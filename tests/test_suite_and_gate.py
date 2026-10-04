import json

from agent_evals.cases import load_cases
from agent_evals.cli import main
from agent_evals.compare import compare
from agent_evals.models import make_model_factory
from agent_evals.report import build_report
from agent_evals.runner import run_suite


def run(cases_dir, world_base, model="scripted", repeat=3):
    cases = load_cases(cases_dir)
    results = run_suite(cases, world_base, make_model_factory(model), repeat=repeat, seed=7)
    return build_report(cases, results, meta={"model": model, "repeat": repeat, "seed": 7})


def test_runs_are_reproducible(cases_dir, world_base):
    # Same seed, same results: the scripted model's mistakes are seeded per run.
    assert run(cases_dir, world_base)["cases"] == run(cases_dir, world_base)["cases"]


def test_baseline_model_passes_most_cases(cases_dir, world_base):
    report = run(cases_dir, world_base, repeat=5)
    assert report["summary"]["pass_rate"] >= 0.9


def test_gate_catches_the_keep_it_short_regression(cases_dir, world_base):
    baseline = run(cases_dir, world_base)
    regressed = run(cases_dir, world_base, model="scripted:keep-it-short")
    result = compare(baseline, regressed, max_regression=0.03)
    assert not result.ok
    assert "status-shipped" in {case for case, _, _ in result.regressed}


def test_gate_passes_an_identical_run(cases_dir, world_base):
    report = run(cases_dir, world_base)
    assert compare(report, report).ok


def test_gate_only_compares_shared_cases():
    def report(cases):
        return {
            "model": "m",
            "repeat": 1,
            "cases": {k: {"runs": 1, "passed": v, "pass_rate": float(v)} for k, v in cases.items()},
        }

    result = compare(report({"a": 1, "b": 1}), report({"a": 1, "new": 0}))
    assert result.ok and result.added == ["new"] and result.removed == ["b"]


def test_cli_run_and_compare(tmp_path, cases_dir):
    out = tmp_path / "report.json"
    world = cases_dir.parent / "world.yaml"
    args = [
        "run",
        "--cases",
        str(cases_dir),
        "--world",
        str(world),
        "--repeat",
        "2",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    report = json.loads(out.read_text())
    assert report["summary"]["runs"] == 2 * report["summary"]["cases"]
    assert main(["compare", str(out), str(out)]) == 0

    bad = tmp_path / "bad.json"
    assert main([*args[:-1], str(bad), "--model", "scripted:keep-it-short"]) == 0
    assert main(["compare", str(out), str(bad)]) == 1
    assert main([*args, "--tag", "no-such-tag"]) == 2
