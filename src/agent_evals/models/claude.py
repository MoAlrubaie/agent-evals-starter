"""Optional real model: Claude via the Anthropic Python SDK.

Install with `pip install -e ".[claude]"` and set ANTHROPIC_API_KEY. Nothing
in the default demo or CI uses this; it exists so you can point the same
eval suite at a real model.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import ModelTurn, ToolCall, Usage

DEFAULT_MODEL = "claude-opus-5-5"


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise SystemExit(
            'The Claude model needs the optional extra: pip install -e ".[claude]"'
        ) from exc
    return anthropic.Anthropic()


class ClaudeModel:
    def __init__(self, model_id: str | None = None, effort: str | None = None):
        self.model_id = model_id or os.environ.get("AGENT_EVALS_MODEL", DEFAULT_MODEL)
        # Effort trades thoroughness for tokens. Set AGENT_EVALS_EFFORT=none for
        # models that don't accept the parameter.
        self.effort = effort or os.environ.get("AGENT_EVALS_EFFORT", "medium")
        self.name = f"claude:{self.model_id}"
        self.client = _client()

    def respond(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": 16000,
            "system": system,
            "tools": tools,
            "messages": to_api_messages(messages),
        }
        if self.effort and self.effort.lower() != "none":
            params["output_config"] = {"effort": self.effort}

        response = self.client.messages.create(**params)

        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, args=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        return ModelTurn(
            text=text,
            tool_calls=calls,
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            stop_reason=response.stop_reason or "end_turn",
            # Echo the full content back unchanged (it may include thinking blocks).
            raw=response.content,
        )


def to_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the harness's neutral messages to Messages API format.

    Tool results that follow one assistant turn go back together in a single
    user message, as the API expects for parallel tool calls.
    """
    api: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            api.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            content = message.get("raw")
            if content is None:  # produced by another model; rebuild the blocks
                content = []
                if message.get("text"):
                    content.append({"type": "text", "text": message["text"]})
                for call in message.get("tool_calls", []):
                    content.append(
                        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
                    )
            api.append({"role": "assistant", "content": content})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": json.dumps(message["content"]),
                "is_error": message["is_error"],
            }
            previous = api[-1] if api else None
            if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                previous["content"].append(block)
            else:
                api.append({"role": "user", "content": [block]})
    return api
