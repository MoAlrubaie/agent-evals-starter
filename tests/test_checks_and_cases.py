import pytest

from agent_evals.agent import ToolCallRecord, Transcript
from agent_evals.cases import Case, CaseError, load_cases
from agent_evals.checks import evaluate, get_path


def make_transcript(calls=(), reply="Done.", steps=2, stop="reply"):
    records = [ToolCallRecord(name, args, {}, False) for name, args in calls]
    return Transcript(user_input="hi", reply=reply, steps=steps, tool_calls=records, stop=stop)


def failed(checks):
    return [c.name for c in checks if not c.passed and not c.skipped]


WORLD = {"orders": {"1004": {"refunded": 0, "status": "delivered"}}, "escalations": [{"id": "E"}]}


def test_passing_case_has_no_failures():
    case = Case(
        id="c",
        input="x",
        expect={
            "tools_called": ["get_order"],
            "tools_not_called": ["issue_refund"],
            "tool_order": ["get_order", "escalate_to_human"],
            "tool_args": {"get_order": {"order_id": "1004"}},
            "final_state": {"orders.1004.refunded": 0},
            "escalated": True,
            "reply_contains": ["follow up"],
            "reply_not_contains": ["refunded"],
            "max_steps": 3,
        },
    )
    transcript = make_transcript(
        [
            ("get_order", {"order_id": "1004"}),
            ("escalate_to_human", {"order_id": "1004", "reason": "r"}),
        ],
        reply="A colleague will follow up.",
        steps=3,
    )
    assert failed(evaluate(case, WORLD, transcript)) == []


def test_each_expectation_can_fail():
    case = Case(
        id="c",
        input="x",
        expect={
            "tools_called": ["get_order"],
            "tools_not_called": ["issue_refund"],
            "tool_order": ["get_order", "issue_refund"],
            "final_state": {"orders.1004.refunded": 950, "orders.9999.status": "x"},
            "escalated": False,
            "reply_contains": ["sorry"],
            "reply_not_contains": ["done"],
            "max_steps": 1,
        },
    )
    transcript = make_transcript(
        [("issue_refund", {"order_id": "1004", "amount": 950, "reason": "r"})]
    )
    assert len(failed(evaluate(case, WORLD, transcript))) == 9


def test_builtin_checks_catch_bad_calls_budget_and_refusals():
    case = Case(id="c", input="x", expect={"max_steps": 10})
    bad = make_transcript([("get_order", {})], stop="max_steps")
    bad.tool_calls[0].invalid = "missing required argument(s): order_id"
    names = failed(evaluate(case, WORLD, bad))
    assert "tool calls match their schemas" in names
    assert "finished within the step budget" in names
    assert "model did not refuse" in failed(evaluate(case, WORLD, make_transcript(stop="refusal")))


def test_judge_checks_are_skipped_without_a_judge_and_used_with_one():
    case = Case(id="c", input="x", judge=["Is it polite?"])
    checks = evaluate(case, WORLD, make_transcript())
    assert [c.skipped for c in checks if c.layer == "judge"] == [True]

    class NoJudge:
        name = "fake"

        def ask(self, question, transcript):
            return False, "it was rude"

    assert failed(evaluate(case, WORLD, make_transcript(), judge=NoJudge())) == ["Is it polite?"]


def test_get_path():
    assert get_path({"a": {"b": [1, {"c": 2}]}}, "a.b.1.c") == (True, 2)
    assert get_path({"a": {}}, "a.missing") == (False, None)


def test_repo_cases_load(cases_dir):
    cases = load_cases(cases_dir)
    assert len(cases) >= 20
    assert len({c.id for c in cases}) == len(cases)
    assert any("smoke" in c.tags for c in cases)


@pytest.mark.parametrize(
    "content, message",
    [
        ("- {id: a, input: x, expect: {tools_calld: [get_order]}}", "unknown expect key"),
        ("- {id: a, input: x}", "at least one expectation"),
        (
            "- {id: a, input: x, expect: {max_steps: 1}}\n"
            "- {id: a, input: y, expect: {max_steps: 1}}",
            "duplicate",
        ),
        ("- {input: x}", "missing 'id'"),
        ("{id: a}", "list of cases"),
    ],
)
def test_case_validation(tmp_path, content, message):
    path = tmp_path / "cases.yaml"
    path.write_text(content)
    with pytest.raises(CaseError, match=message):
        load_cases(path)
