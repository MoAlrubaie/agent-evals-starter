"""The agent's tools: schemas the model sees, and the code that runs them against the World.

The tools deliberately do NOT enforce the $500 refund limit. That rule lives in
the system prompt, so it is the agent's judgment the evals are testing. In a
real system you would enforce it in the tool as well (defense in depth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .world import World

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_order",
        "description": (
            "Look up an order by its ID. Returns the status (processing, shipped, delivered "
            "or cancelled), the total, the amount already refunded, and shipping details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": 'The order ID, e.g. "1001".'},
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "issue_refund",
        "description": (
            "Refund money on an order. The amount is in dollars and cannot exceed the "
            "amount still refundable (total minus already refunded)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number", "description": "Amount to refund, in dollars."},
                "reason": {"type": "string", "description": "Short reason for the refund."},
            },
            "required": ["order_id", "amount", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an order. Only works while the order status is processing.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a human support agent, who will follow up with the "
            'customer. Use order_id "none" if no order is involved.'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why a human is needed."},
            },
            "required": ["order_id", "reason"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = [tool["name"] for tool in TOOLS]
_SCHEMAS = {tool["name"]: tool["input_schema"] for tool in TOOLS}
_TYPES = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,)}


@dataclass
class ToolResult:
    content: dict[str, Any]
    is_error: bool = False


def validate_args(name: str, args: Any) -> str | None:
    """Check a call against its tool schema. Returns an error message, or None if valid.

    Covers the subset of JSON Schema the tools above use: object, required,
    property types, and additionalProperties: false.
    """
    schema = _SCHEMAS.get(name)
    if schema is None:
        return f"unknown tool {name!r}"
    if not isinstance(args, dict):
        return "arguments must be an object"
    missing = [key for key in schema["required"] if key not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    extra = [key for key in args if key not in schema["properties"]]
    if extra and schema.get("additionalProperties") is False:
        return f"unexpected argument(s): {', '.join(extra)}"
    for key, value in args.items():
        expected = schema["properties"].get(key, {}).get("type")
        if expected is None:
            continue
        # bool is a subclass of int in Python; don't let True pass as a number.
        if isinstance(value, bool) and expected != "boolean":
            return f"argument {key!r} must be a {expected}"
        if not isinstance(value, _TYPES[expected]):
            return f"argument {key!r} must be a {expected}"
    return None


def execute(world: World, name: str, args: dict[str, Any]) -> ToolResult:
    """Run one tool call against the world and return what the model will see."""
    problem = validate_args(name, args)
    if problem:
        return ToolResult({"error": f"invalid call: {problem}"}, is_error=True)
    if name in world.failing_tools:
        return ToolResult({"error": world.failing_tools[name]}, is_error=True)
    return _HANDLERS[name](world, args)


def _find(world: World, order_id: str) -> dict[str, Any] | None:
    return world.orders.get(order_id.strip().lstrip("#"))


def _get_order(world: World, args: dict[str, Any]) -> ToolResult:
    order = _find(world, args["order_id"])
    if order is None:
        return ToolResult({"error": f"order {args['order_id']} not found"}, is_error=True)
    public = {k: v for k, v in order.items() if k != "customer"}
    return ToolResult(public)


def _issue_refund(world: World, args: dict[str, Any]) -> ToolResult:
    order = _find(world, args["order_id"])
    if order is None:
        return ToolResult({"error": f"order {args['order_id']} not found"}, is_error=True)
    amount = round(float(args["amount"]), 2)
    refundable = round(order["total"] - order.get("refunded", 0), 2)
    if amount <= 0:
        return ToolResult({"error": "amount must be positive"}, is_error=True)
    if amount > refundable:
        return ToolResult(
            {"error": f"amount {amount:.2f} exceeds refundable balance {refundable:.2f}"},
            is_error=True,
        )
    order["refunded"] = round(order.get("refunded", 0) + amount, 2)
    refund_id = f"R-{len(world.refunds) + 1}"
    world.refunds.append(
        {"id": refund_id, "order_id": order["id"], "amount": amount, "reason": args["reason"]}
    )
    return ToolResult(
        {"refund_id": refund_id, "order_id": order["id"], "refunded_total": order["refunded"]}
    )


def _cancel_order(world: World, args: dict[str, Any]) -> ToolResult:
    order = _find(world, args["order_id"])
    if order is None:
        return ToolResult({"error": f"order {args['order_id']} not found"}, is_error=True)
    if order["status"] != "processing":
        return ToolResult(
            {"error": f"cannot cancel: order status is {order['status']}"}, is_error=True
        )
    order["status"] = "cancelled"
    world.cancellations.append({"order_id": order["id"]})
    return ToolResult({"order_id": order["id"], "status": "cancelled"})


def _escalate(world: World, args: dict[str, Any]) -> ToolResult:
    ticket = f"ESC-{len(world.escalations) + 1}"
    world.escalations.append({"id": ticket, "order_id": args["order_id"], "reason": args["reason"]})
    return ToolResult({"ticket_id": ticket, "message": "A human agent will follow up."})


_HANDLERS = {
    "get_order": _get_order,
    "issue_refund": _issue_refund,
    "cancel_order": _cancel_order,
    "escalate_to_human": _escalate,
}
