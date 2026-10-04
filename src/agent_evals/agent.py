"""The agent under test: a plain tool-calling loop.

The loop is intentionally boring. What changes between experiments is the
model, the system prompt and the tools, and that's what the evals measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import tools as toolbox
from .models.base import Model, Usage
from .prompts import SYSTEM_PROMPT
from .world import World


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    is_error: bool
    # None when the call matched its schema, otherwise why it didn't.
    invalid: str | None = None


@dataclass
class Transcript:
    user_input: str
    reply: str = ""
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # "reply" (finished normally), "max_steps", "refusal", or "error"
    stop: str = "reply"
    messages: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable transcript, for -v output and for the LLM judge."""
        lines = [f"user: {self.user_input}"]
        for call in self.tool_calls:
            status = "error" if call.is_error else "ok"
            lines.append(f"tool: {call.name}({_fmt_args(call.args)}) -> {status} {call.result}")
        lines.append(f"agent: {self.reply or '(no reply)'}")
        if self.stop != "reply":
            lines.append(f"[stopped: {self.stop}]")
        return "\n".join(lines)


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def run_agent(
    model: Model,
    world: World,
    user_input: str,
    *,
    max_steps: int = 10,
    system: str = SYSTEM_PROMPT,
) -> Transcript:
    """Run one conversation turn to completion. A step is one model response."""
    transcript = Transcript(user_input=user_input)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
    transcript.messages = messages

    for _ in range(max_steps):
        turn = model.respond(system, messages, toolbox.TOOLS)
        transcript.steps += 1
        transcript.usage.add(turn.usage)
        messages.append(
            {"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls, "raw": turn.raw}
        )

        if not turn.tool_calls:
            transcript.reply = turn.text.strip()
            transcript.stop = "refusal" if turn.stop_reason == "refusal" else "reply"
            return transcript

        for call in turn.tool_calls:
            result = toolbox.execute(world, call.name, call.args)
            transcript.tool_calls.append(
                ToolCallRecord(
                    name=call.name,
                    args=call.args,
                    result=result.content,
                    is_error=result.is_error,
                    invalid=toolbox.validate_args(call.name, call.args),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "args": call.args,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )

    transcript.stop = "max_steps"
    # Out of budget mid-task: keep whatever the agent last said, if anything.
    said = [m["text"] for m in messages if m["role"] == "assistant" and m["text"]]
    transcript.reply = said[-1].strip() if said else ""
    return transcript
