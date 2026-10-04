"""The model interface the agent loop talks to.

Messages use a small provider-neutral format so the same loop can drive the
scripted fake model and a real LLM:

    {"role": "user", "content": "Where is order 1002?"}
    {"role": "assistant", "text": "...", "tool_calls": [ToolCall, ...], "raw": <provider data>}
    {"role": "tool", "tool_call_id": "...", "name": "get_order", "args": {...},
     "content": {...}, "is_error": False}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelTurn:
    """One response from the model: some text, and zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # "end_turn", "tool_use", "refusal", "max_tokens", ... (provider's stop reason)
    stop_reason: str = "end_turn"
    # Provider-specific payload the provider needs echoed back on the next turn.
    raw: Any = None


class Model(Protocol):
    name: str

    def respond(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...
