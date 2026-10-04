"""Model factory: turns a `--model` string into a model instance."""

from __future__ import annotations

from collections.abc import Callable

from .base import Model, ModelTurn, ToolCall, Usage
from .scripted import BEHAVIORS, ScriptedModel

__all__ = ["Model", "ModelTurn", "ToolCall", "Usage", "make_model_factory", "MODEL_HELP"]

MODEL_HELP = (
    "scripted (default, no API key), "
    + ", ".join(f"scripted:{b}" for b in BEHAVIORS if b != "baseline")
    + ", claude, or claude:<model-id>"
)


def make_model_factory(spec: str) -> Callable[[int], Model]:
    """Return a function that builds a fresh model for one run, given that run's seed."""
    kind, _, variant = spec.partition(":")
    if kind == "scripted":
        behavior = variant or "baseline"
        ScriptedModel(behavior)  # fail fast on a typo before any run starts
        return lambda seed: ScriptedModel(behavior, seed=seed)
    if kind == "claude":
        from .claude import ClaudeModel

        model = ClaudeModel(model_id=variant or None)
        return lambda seed: model  # stateless across runs; the seed doesn't apply
    raise ValueError(f"unknown model {spec!r}; expected {MODEL_HELP}")
