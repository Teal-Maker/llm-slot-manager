"""Tests for SlotManager construction and validation."""

import pytest

from llm_slot_manager import Priority, SlotManager


def test_valid_construction() -> None:
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={"large": {"slots": 4, "reserved": 1}},
    )
    assert mgr is not None


def test_empty_tiers_raises() -> None:
    with pytest.raises(ValueError, match="At least one tier"):
        SlotManager(redis_url="redis://localhost:6379/0", tiers={})


def test_tier_missing_slots_raises() -> None:
    with pytest.raises(ValueError, match="missing required key 'slots'"):
        SlotManager(
            redis_url="redis://localhost:6379/0",
            tiers={"large": {"reserved": 1}},
        )


def test_tier_slots_zero_raises() -> None:
    with pytest.raises(ValueError, match="slots must be >= 1"):
        SlotManager(
            redis_url="redis://localhost:6379/0",
            tiers={"large": {"slots": 0}},
        )


def test_tier_reserved_negative_raises() -> None:
    with pytest.raises(ValueError, match="reserved must be >= 0"):
        SlotManager(
            redis_url="redis://localhost:6379/0",
            tiers={"large": {"slots": 4, "reserved": -1}},
        )


def test_tier_reserved_gte_slots_raises() -> None:
    with pytest.raises(ValueError, match="reserved .* must be < slots"):
        SlotManager(
            redis_url="redis://localhost:6379/0",
            tiers={"large": {"slots": 4, "reserved": 4}},
        )


def test_multiple_tiers_valid() -> None:
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={
            "large": {"slots": 4, "reserved": 1},
            "small": {"slots": 6, "reserved": 1},
            "tiny": {"slots": 2},
        },
    )
    assert mgr is not None


def test_default_priority_timeouts() -> None:
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={"large": {"slots": 4}},
    )
    # Defaults should match the documented values
    assert mgr._priority_timeouts[Priority.CRITICAL] == 120
    assert mgr._priority_timeouts[Priority.HIGH] == 60
    assert mgr._priority_timeouts[Priority.MEDIUM] == 30
    assert mgr._priority_timeouts[Priority.LOW] == 10


def test_custom_priority_timeouts() -> None:
    custom = {Priority.CRITICAL: 5, Priority.HIGH: 3, Priority.MEDIUM: 2, Priority.LOW: 1}
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={"large": {"slots": 4}},
        priority_timeouts=custom,
    )
    assert mgr._priority_timeouts[Priority.LOW] == 1


def test_worker_id_is_unique() -> None:
    mgr1 = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={"large": {"slots": 4}},
    )
    mgr2 = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={"large": {"slots": 4}},
    )
    assert mgr1._worker_id != mgr2._worker_id


def test_enable_metrics_without_prometheus_raises() -> None:
    """When prometheus-client is not installed, enable_metrics raises ImportError."""
    import unittest.mock
    with unittest.mock.patch.dict("sys.modules", {"prometheus_client": None}):
        with pytest.raises(ImportError, match="prometheus-client"):
            SlotManager(
                redis_url="redis://localhost:6379/0",
                tiers={"large": {"slots": 4}},
                enable_metrics=True,
            )
