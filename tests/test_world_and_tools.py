from agent_evals.tools import execute, validate_args
from agent_evals.world import World, deep_merge


def test_deep_merge_patches_nested_keys_and_deletes_with_none():
    base = {"orders": {"1": {"status": "processing", "total": 10}, "2": {"status": "shipped"}}}
    merged = deep_merge(base, {"orders": {"1": {"status": "shipped"}, "2": None}})
    assert merged == {"orders": {"1": {"status": "shipped", "total": 10}}}
    assert base["orders"]["1"]["status"] == "processing"  # the base is not mutated


def test_world_build_normalizes_integer_order_ids():
    world = World.build({"orders": {1001: {"status": "processing", "total": 5}}})
    assert world.orders["1001"]["id"] == "1001"


def test_each_build_is_independent(world_base):
    first = World.build(world_base)
    execute(first, "cancel_order", {"order_id": "1007"})
    assert World.build(world_base).orders["1007"]["status"] == "processing"


def test_validate_args():
    assert validate_args("get_order", {"order_id": "1001"}) is None
    assert "missing" in validate_args("get_order", {})
    assert "unexpected" in validate_args("get_order", {"order_id": "1", "x": 1})
    assert "number" in validate_args(
        "issue_refund", {"order_id": "1", "amount": "5", "reason": "r"}
    )
    assert "number" in validate_args(
        "issue_refund", {"order_id": "1", "amount": True, "reason": "r"}
    )
    assert "unknown tool" in validate_args("delete_everything", {})


def test_refund_cannot_exceed_refundable_balance(world_base):
    world = World.build(world_base)
    result = execute(world, "issue_refund", {"order_id": "1006", "amount": 300, "reason": "r"})
    assert result.is_error
    assert world.orders["1006"]["refunded"] == 450


def test_refund_updates_state(world_base):
    world = World.build(world_base)
    result = execute(world, "issue_refund", {"order_id": "1003", "amount": 40, "reason": "r"})
    assert not result.is_error
    assert world.snapshot()["orders"]["1003"]["refunded"] == 40
    assert world.refunds[0]["amount"] == 40


def test_cancel_only_while_processing(world_base):
    world = World.build(world_base)
    assert execute(world, "cancel_order", {"order_id": "1002"}).is_error
    assert not execute(world, "cancel_order", {"order_id": "1007"}).is_error
    assert world.orders["1007"]["status"] == "cancelled"


def test_get_order_hides_customer_name_but_unknown_order_is_an_error(world_base):
    world = World.build(world_base)
    assert "customer" not in execute(world, "get_order", {"order_id": "1001"}).content
    assert execute(world, "get_order", {"order_id": "9999"}).is_error


def test_failing_tools_simulate_outages(world_base):
    world = World.build(world_base, {"failing_tools": {"get_order": "timeout"}})
    result = execute(world, "get_order", {"order_id": "1001"})
    assert result.is_error and result.content == {"error": "timeout"}
