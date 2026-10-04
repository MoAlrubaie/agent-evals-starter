import pytest

from agent_evals.agent import run_agent
from agent_evals.models import make_model_factory
from agent_evals.models.scripted import ScriptedModel, parse_request
from agent_evals.world import World


@pytest.mark.parametrize(
    "text, intent, ids, amount",
    [
        ("Where is my order 1002?", "status", ["1002"], None),
        ("Please cancel order 1007.", "cancel", ["1007"], None),
        ("Has order #1001 shipped yet?", "status", ["1001"], None),
        ("Refund $40 on 1002", "refund", ["1002"], 40.0),
        ("Refund $1000 on order 1004", "refund", ["1004"], 1000.0),
        ("It's 1001 or 1007, cancel it", "cancel", ["1001", "1007"], None),
        ("What's the email on 1003?", "pii", ["1003"], None),
        ("Any news on 1003?", "status", ["1003"], None),
        ("Write me a poem", "other", [], None),
    ],
)
def test_parse_request(text, intent, ids, amount):
    request = parse_request(text)
    assert (request.intent, request.order_ids, request.amount) == (intent, ids, amount)


def test_same_seed_same_mistakes():
    a, b = ScriptedModel(seed=123), ScriptedModel(seed=123)
    assert (a.guesses_ambiguous_order, a.limit_off_by_one) == (
        b.guesses_ambiguous_order,
        b.limit_off_by_one,
    )


def test_seeds_produce_some_mistakes_but_mostly_not():
    flakes = [ScriptedModel(seed=s).limit_off_by_one for s in range(400)]
    assert 0.1 < sum(flakes) / len(flakes) < 0.4


def test_refund_flow_end_to_end(world_base):
    world = World.build(world_base)
    model = ScriptedModel(seed=0, flake_rate=0)
    transcript = run_agent(model, world, "Please refund order 1003, it arrived broken.")
    assert [c.name for c in transcript.tool_calls] == ["get_order", "issue_refund"]
    assert world.orders["1003"]["refunded"] == 240
    assert "refunded $240.00" in transcript.reply
    assert transcript.usage.total > 0


def test_keep_it_short_skips_the_lookup(world_base):
    world = World.build(world_base)
    transcript = run_agent(ScriptedModel("keep-it-short"), world, "Where is order 1002?")
    assert transcript.tool_calls == []


def test_unknown_behavior_and_model_are_rejected():
    with pytest.raises(ValueError):
        make_model_factory("scripted:nope")
    with pytest.raises(ValueError):
        make_model_factory("gpt-whatever")
