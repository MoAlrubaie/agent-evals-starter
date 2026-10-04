"""The Claude adapter, tested without network access or an API key."""

from types import SimpleNamespace

from agent_evals.agent import run_agent
from agent_evals.models import claude
from agent_evals.models.base import ToolCall
from agent_evals.world import World


def test_tool_results_are_grouped_into_one_user_message():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "text": "Checking both.",
            "tool_calls": [
                ToolCall("a", "get_order", {"order_id": "1"}),
                ToolCall("b", "get_order", {"order_id": "2"}),
            ],
            "raw": None,
        },
        {
            "role": "tool",
            "tool_call_id": "a",
            "name": "get_order",
            "args": {},
            "content": {"x": 1},
            "is_error": False,
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "name": "get_order",
            "args": {},
            "content": {"error": "no"},
            "is_error": True,
        },
    ]
    api = claude.to_api_messages(messages)
    assert [m["role"] for m in api] == ["user", "assistant", "user"]
    assert [b["type"] for b in api[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in api[2]["content"]] == ["a", "b"]
    assert api[2]["content"][1]["is_error"] is True


class FakeMessages:
    """Replays canned responses and records the requests it receives."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def create(self, **params):
        self.requests.append(params)
        return self.responses.pop(0)


def block(**fields):
    return SimpleNamespace(**fields)


def response(content, stop_reason):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_claude_model_drives_the_agent_loop(monkeypatch, world_base):
    fake = FakeMessages(
        [
            response(
                [block(type="tool_use", id="t1", name="get_order", input={"order_id": "1002"})],
                "tool_use",
            ),
            response([block(type="text", text="Order 1002 has shipped.")], "end_turn"),
        ]
    )
    monkeypatch.setattr(claude, "_client", lambda: SimpleNamespace(messages=fake))
    model = claude.ClaudeModel(model_id="test-model", effort="low")

    transcript = run_agent(model, World.build(world_base), "Where is 1002?")

    assert transcript.reply == "Order 1002 has shipped."
    assert transcript.usage.total == 30
    first, second = fake.requests
    assert first["model"] == "test-model" and first["output_config"] == {"effort": "low"}
    # The assistant turn is echoed back unchanged, followed by the tool result.
    assert second["messages"][1]["content"][0].id == "t1"
    assert second["messages"][2]["content"][0]["tool_use_id"] == "t1"


def test_refusal_is_recorded(monkeypatch, world_base):
    fake = FakeMessages([response([], "refusal")])
    monkeypatch.setattr(claude, "_client", lambda: SimpleNamespace(messages=fake))
    transcript = run_agent(claude.ClaudeModel(effort="none"), World.build(world_base), "hi")
    assert transcript.stop == "refusal"
    assert "output_config" not in fake.requests[0]
