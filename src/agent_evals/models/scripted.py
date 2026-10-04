"""A scripted stand-in for an LLM, so the harness runs with no API key and costs nothing.

It reads the customer's message with a few regular expressions and follows the
support policy step by step, calling the same tools a real model would. It is
not smart, and it doesn't need to be: its job is to exercise the harness.

Two things make it useful for demonstrating evals:

* **Seeded non-determinism.** Like a real model, it occasionally gets a case
  wrong. Each run draws its "mistakes" from a seeded RNG, so results vary
  between runs but are reproducible for a given seed. That is why the harness
  reports pass *rates*.
* **Behaviors.** `scripted:keep-it-short` and `scripted:eager-refunder` are
  deliberately regressed variants, so you can watch the CI gate catch a
  regression (see `make demo-regression`).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .base import ModelTurn, ToolCall, Usage

REFUND_LIMIT = 500

BEHAVIORS = {
    "baseline": "Follows the policy, with occasional seeded mistakes.",
    "keep-it-short": "Someone added 'keep answers short': it stops looking orders up "
    "before answering status questions.",
    "eager-refunder": "Ignores the $500 refund limit.",
}

_AMOUNT = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
# Four digits, not part of a longer number or a price like 1002.50.
_ORDER = re.compile(r"(?<![\d.$])#?(\d{4})(?!\d|\.\d)")
_INTENTS = [
    ("pii", re.compile(r"\b(e-?mail|phone|address)\b", re.I)),
    ("refund", re.compile(r"\brefund|money back", re.I)),
    ("cancel", re.compile(r"\bcancel", re.I)),
    ("status", re.compile(r"\b(where|status|shipped|arriv|track|news|eta|delivered)", re.I)),
]

OFF_TOPIC = "Sorry, I can only help with orders from this store: status, refunds and cancellations."


@dataclass
class Request:
    intent: str  # pii | refund | cancel | status | other
    order_ids: list[str] = field(default_factory=list)
    amount: float | None = None


def parse_request(text: str) -> Request:
    amounts = _AMOUNT.findall(text)
    without_amounts = _AMOUNT.sub(" ", text)
    order_ids = list(dict.fromkeys(_ORDER.findall(without_amounts)))
    intent = next((name for name, pattern in _INTENTS if pattern.search(text)), "other")
    if intent == "other" and order_ids:
        intent = "status"  # "Any news on 1003?" style messages
    return Request(intent, order_ids, float(amounts[0]) if amounts else None)


class ScriptedModel:
    def __init__(self, behavior: str = "baseline", seed: int = 0, flake_rate: float = 0.25):
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown scripted behavior {behavior!r}; try {', '.join(BEHAVIORS)}")
        self.behavior = behavior
        self.name = "scripted" if behavior == "baseline" else f"scripted:{behavior}"
        rng = random.Random(seed)
        # Mistakes are drawn once per run so the model is consistent within a conversation.
        self.guesses_ambiguous_order = rng.random() < flake_rate
        self.limit_off_by_one = rng.random() < flake_rate
        self._call_count = 0

    # -- Model protocol ----------------------------------------------------

    def respond(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        request = parse_request(messages[0]["content"])
        results = [m for m in messages if m["role"] == "tool"]
        action = self._decide(request, results)
        turn = action if isinstance(action, ModelTurn) else self._call(*action)
        turn.usage = _estimate_usage(system, messages, tools, turn)
        return turn

    # -- policy ------------------------------------------------------------

    def _decide(self, req: Request, results: list[dict[str, Any]]):
        if req.intent == "pii":
            return _reply(
                "I can't share personal details stored on an order. The account holder "
                "can see them in their account settings."
            )
        if not req.order_ids:
            if req.intent == "other":
                return _reply(OFF_TOPIC)
            return _reply("Happy to help. What's your order number?")
        if len(req.order_ids) > 1 and not self.guesses_ambiguous_order:
            listed = " and ".join(req.order_ids)
            return _reply(f"You mentioned orders {listed}. Which one do you mean?")

        oid = req.order_ids[0]
        if self.behavior == "keep-it-short" and req.intent == "status":
            return _reply("Your order is on its way!")

        lookup = _last(results, "get_order")
        if lookup is None:
            return ("get_order", {"order_id": oid})
        if lookup["is_error"]:
            if "not found" in lookup["content"].get("error", ""):
                return _reply(f"I couldn't find order {oid}. Could you double-check the number?")
            return self._escalate(
                results,
                oid,
                "order lookup failed",
                f"I couldn't look up order {oid} right now, so I've passed it to a colleague "
                "who will get back to you.",
            )

        order = lookup["content"]
        if req.intent == "refund":
            return self._refund(req, order, results)
        if req.intent == "cancel":
            return self._cancel(order, results)
        return _reply(_describe(order))

    def _refund(self, req: Request, order: dict[str, Any], results: list[dict[str, Any]]):
        oid = order["id"]
        refundable = round(order["total"] - order.get("refunded", 0), 2)
        if refundable <= 0:
            return _reply(f"Order {oid} has already been refunded in full.")
        amount = req.amount if req.amount is not None else refundable
        if amount > refundable:
            return _reply(
                f"Only ${refundable:.2f} is still refundable on order {oid}, so I can't refund "
                f"${amount:.2f}. Would you like me to refund ${refundable:.2f} instead?"
            )
        over_limit = amount >= REFUND_LIMIT if self.limit_off_by_one else amount > REFUND_LIMIT
        if over_limit and self.behavior != "eager-refunder":
            return self._escalate(
                results,
                oid,
                f"refund of ${amount:.2f} is over the ${REFUND_LIMIT} limit",
                f"Refunds over ${REFUND_LIMIT} need a person to approve them, so I've passed "
                f"order {oid} to a colleague who will follow up shortly.",
            )
        refund = _last(results, "issue_refund")
        if refund is None:
            return (
                "issue_refund",
                {"order_id": oid, "amount": amount, "reason": "customer request"},
            )
        if refund["is_error"]:
            return self._escalate(
                results,
                oid,
                "refund failed",
                f"I couldn't process the refund on order {oid} right now, so I've passed it "
                "to a colleague who will follow up.",
            )
        return _reply(f"Done: I've refunded ${amount:.2f} on order {oid}.")

    def _cancel(self, order: dict[str, Any], results: list[dict[str, Any]]):
        oid, status = order["id"], order["status"]
        if status == "cancelled":
            return _reply(f"Order {oid} is already cancelled.")
        if status != "processing":
            already = "been delivered" if status == "delivered" else status
            return _reply(
                f"Order {oid} has already {already}, so it can't be cancelled. You can return "
                "it for a refund once you have it."
            )
        cancel = _last(results, "cancel_order")
        if cancel is None:
            return ("cancel_order", {"order_id": oid})
        if cancel["is_error"]:
            return self._escalate(
                results,
                oid,
                "cancellation failed",
                f"I couldn't cancel order {oid} right now, so I've passed it to a colleague.",
            )
        return _reply(f"Done: order {oid} is cancelled.")

    def _escalate(self, results, oid: str, reason: str, message: str):
        if _last(results, "escalate_to_human") is None:
            return ("escalate_to_human", {"order_id": oid, "reason": reason})
        return _reply(message)

    def _call(self, name: str, args: dict[str, Any]) -> ModelTurn:
        self._call_count += 1
        call = ToolCall(id=f"call_{self._call_count}", name=name, args=args)
        return ModelTurn(tool_calls=[call], stop_reason="tool_use")


def _reply(text: str) -> ModelTurn:
    return ModelTurn(text=text, stop_reason="end_turn")


def _last(results: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    matches = [r for r in results if r["name"] == tool]
    return matches[-1] if matches else None


def _describe(order: dict[str, Any]) -> str:
    oid, status = order["id"], order["status"]
    if status == "processing":
        ships_in = order.get("ships_in", "a few days")
        return f"Order {oid} is still processing and should ship within {ships_in}."
    if status == "shipped":
        return (
            f"Order {oid} has shipped (tracking {order.get('tracking', 'n/a')}) and should "
            f"arrive {order.get('eta', 'soon')}."
        )
    if status == "delivered":
        return f"Order {oid} was delivered on {order.get('delivered_on', 'a recent date')}."
    return f"Order {oid} is {status}."


def _estimate_usage(system, messages, tools, turn: ModelTurn) -> Usage:
    """Rough token estimate (~4 characters per token), so cost tracking has numbers to show."""
    prompt = system + json.dumps(tools) + json.dumps(messages, default=str)
    output = turn.text + json.dumps([c.args for c in turn.tool_calls])
    return Usage(input_tokens=len(prompt) // 4, output_tokens=max(1, len(output) // 4))
