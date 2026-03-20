"""Tests for get_occupancy() and get_utilization()."""

from __future__ import annotations

import pytest

from llm_slot_manager import Priority, SlotManager


def test_occupancy_empty(manager: SlotManager) -> None:
    occupied, total = manager.get_occupancy("large")
    assert occupied == 0
    assert total == 4


def test_occupancy_with_slots_held(manager: SlotManager) -> None:
    redis_client = manager._redis_client

    with manager.slot("large", priority=Priority.HIGH):
        occupied, total = manager.get_occupancy("large")
        assert occupied == 1
        assert total == 4


def test_occupancy_all_slots_held(manager: SlotManager) -> None:
    redis_client = manager._redis_client

    with manager.slot("large", priority=Priority.HIGH):
        with manager.slot("large", priority=Priority.HIGH):
            with manager.slot("large", priority=Priority.HIGH):
                with manager.slot("large", priority=Priority.HIGH):
                    occupied, total = manager.get_occupancy("large")
                    assert occupied == 4
                    assert total == 4


def test_occupancy_after_release(manager: SlotManager) -> None:
    with manager.slot("large", priority=Priority.MEDIUM):
        pass

    occupied, total = manager.get_occupancy("large")
    assert occupied == 0
    assert total == 4


def test_utilization_empty(manager: SlotManager) -> None:
    util = manager.get_utilization("large")
    assert util == 0.0


def test_utilization_half_full(manager: SlotManager) -> None:
    with manager.slot("large", priority=Priority.HIGH):
        with manager.slot("large", priority=Priority.HIGH):
            util = manager.get_utilization("large")
            assert util == pytest.approx(0.5)


def test_utilization_full(manager: SlotManager) -> None:
    redis_client = manager._redis_client
    # Fill all 4 slots directly
    for i in range(4):
        redis_client.set(f"{manager._key_prefix}:large:{i}", "other-worker", ex=60)

    util = manager.get_utilization("large")
    assert util == pytest.approx(1.0)


def test_occupancy_unknown_tier_raises(manager: SlotManager) -> None:
    with pytest.raises(KeyError, match="unknown"):
        manager.get_occupancy("unknown")


def test_utilization_unknown_tier_raises(manager: SlotManager) -> None:
    with pytest.raises(KeyError, match="unknown"):
        manager.get_utilization("unknown")


def test_occupancy_returns_zero_when_redis_unavailable() -> None:
    mgr = SlotManager(
        redis_url="redis://localhost:19997/0",
        tiers={"large": {"slots": 4}},
        fail_open=True,
    )
    occupied, total = mgr.get_occupancy("large")
    assert occupied == 0
    assert total == 4


def test_utilization_returns_zero_when_redis_unavailable() -> None:
    mgr = SlotManager(
        redis_url="redis://localhost:19997/0",
        tiers={"large": {"slots": 4}},
        fail_open=True,
    )
    util = mgr.get_utilization("large")
    assert util == 0.0


def test_separate_tiers_are_independent(manager: SlotManager) -> None:
    """Slots in one tier do not affect occupancy counts in another."""
    with manager.slot("large", priority=Priority.HIGH):
        small_occ, small_total = manager.get_occupancy("small")
        large_occ, large_total = manager.get_occupancy("large")

    assert small_occ == 0
    assert small_total == 2
    assert large_occ == 1
    assert large_total == 4
