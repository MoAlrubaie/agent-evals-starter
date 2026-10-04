"""LLM-as-judge: the thin top layer of the pyramid.

Each judge question is a narrow yes/no question about one transcript ("Does
the reply tell the customer a person will follow up?"), not a 1-10 score.
The judge model is pinned and the prompt lives here, under version control,
so scores stay comparable over time.

Judge checks only run with `--judge claude`. Without a judge they are
reported as skipped, never as passed.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from .agent import Transcript

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"

JUDGE_SYSTEM = """\
You grade transcripts of a customer support agent. You will get one transcript and one \
yes/no question about it. Answer strictly from what the transcript shows. If the \
transcript doesn't clearly support "yes", answer "no". Give a one-sentence reason first.\
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "answer": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["reason", "answer"],
    "additionalProperties": False,
}


class Judge(Protocol):
    name: str

    def ask(self, question: str, transcript: Transcript) -> tuple[bool, str]: ...


class ClaudeJudge:
    def __init__(self, model_id: str | None = None):
        from .models.claude import _client

        self.model_id = model_id or os.environ.get("AGENT_EVALS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self.name = f"claude:{self.model_id}"
        self.client = _client()

    def ask(self, question: str, transcript: Transcript) -> tuple[bool, str]:
        prompt = f"<transcript>\n{transcript.render()}\n</transcript>\n\nQuestion: {question}"
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=16000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        )
        if response.stop_reason == "refusal":
            return False, "judge refused to answer"
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            verdict = json.loads(text)
        except json.JSONDecodeError:
            return False, f"judge returned unparseable output: {text[:80]!r}"
        return verdict.get("answer") == "yes", str(verdict.get("reason", ""))


def make_judge(spec: str | None) -> Judge | None:
    if not spec or spec == "none":
        return None
    kind, _, variant = spec.partition(":")
    if kind == "claude":
        return ClaudeJudge(model_id=variant or None)
    raise ValueError(f"unknown judge {spec!r}; expected none, claude or claude:<model-id>")
