"""The simulated world the agent acts on: an in-memory order database.

Every eval run gets a fresh World built from `evals/world.yaml` plus the case's
`setup` patch, so runs never leak state into each other. After the run, the
checks inspect `World.snapshot()` to see what the agent actually did.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


def deep_merge(base: dict, patch: dict) -> dict:
    """Return `base` with `patch` merged in. A `None` value in the patch deletes the key."""
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _stringify_keys(value: Any) -> Any:
    """YAML reads `1001:` as an int key; order IDs are strings everywhere else."""
    if isinstance(value, dict):
        return {str(k): _stringify_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(v) for v in value]
    return value


@dataclass
class World:
    orders: dict[str, dict[str, Any]]
    refunds: list[dict[str, Any]] = field(default_factory=list)
    cancellations: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[dict[str, Any]] = field(default_factory=list)
    # Tool name -> error message. Lets a case simulate a timeout or an outage.
    failing_tools: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, base: dict | None, setup: dict | None = None) -> World:
        data = deep_merge(_stringify_keys(base or {}), _stringify_keys(setup or {}))
        orders = {
            order_id: {"id": order_id, **fields}
            for order_id, fields in (data.get("orders") or {}).items()
        }
        return cls(orders=orders, failing_tools=dict(data.get("failing_tools") or {}))

    def snapshot(self) -> dict[str, Any]:
        return {
            "orders": copy.deepcopy(self.orders),
            "refunds": copy.deepcopy(self.refunds),
            "cancellations": copy.deepcopy(self.cancellations),
            "escalations": copy.deepcopy(self.escalations),
        }
