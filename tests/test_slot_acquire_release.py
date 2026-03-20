"""Tests for sync slot acquisition and release via SlotManager.slot()."""

from __future__ import annotations

import threading
import time

import fakeredis
import pytest

from llm_slot_manager import Priority, SlotManager, SlotUnavailable


def test_slot_acquired_and_released(manager: SlotManager) -> None:
    """Slot key is present in Redis while held and absent after release."""
    redis_client = manager._redis_client

    with manager.slot("large", priority=Priority.HIGH):
        keys = redis_client.keys(f"{manager._key_prefix}:large:*")
        assert len(keys) == 1

    keys = redis_client.keys(f"{manager._key_prefix}:large:*")
    assert len(keys) == 0


def test_multiple_slots_can_be_held_simultaneously(manager: SlotManager) -> None:
    """Several concurrent acquisitions up to the slot limit all succeed."""
    redis_client = manager._redis_client

    with manager.slot("large", priority=Priority.HIGH):
        with manager.slot("large", priority=Priority.HIGH):
            with manager.slot("large", priority=Priority.HIGH):
                keys = redis_client.keys(f"{manager._key_prefix}:large:*")
                assert len(keys) == 3

    keys = redis_client.keys(f"{manager._key_prefix}:large:*")
    assert len(keys) == 0


def test_slot_released_on_exception(manager: SlotManager) -> None:
    """Slot is released even when the body raises an exception."""
    redis_client = manager._redis_client

    with pytest.raises(RuntimeError):
        with manager.slot("large", priority=Priority.HIGH):
            raise RuntimeError("body error")

    keys = redis_client.keys(f"{manager._key_prefix}:large:*")
    assert len(keys) == 0


def test_unknown_tier_raises_key_error(manager: SlotManager) -> None:
    with pytest.raises(KeyError, match="unknown-tier"):
        with manager.slot("unknown-tier"):
            pass


def test_low_priority_blocked_by_reserved_slots(
    fail_closed_manager: SlotManager,
) -> None:
    """LOW priority cannot acquire when all unreserved slots are full."""
    mgr = fail_closed_manager
    redis_client = mgr._redis_client

    # "large" tier has 4 slots and 1 reserved → 3 unreserved for LOW
    # Fill all 3 unreserved slots directly
    for i in range(3):
        redis_client.set(f"{mgr._key_prefix}:large:{i}", "other-worker", ex=60)

    with pytest.raises(SlotUnavailable) as exc_info:
        with mgr.slot("large", priority=Priority.LOW, timeout=1):
            pass

    assert exc_info.value.tier == "large"
    assert exc_info.value.priority == Priority.LOW


def test_high_priority_can_use_reserved_slot(
    fail_closed_manager: SlotManager,
) -> None:
    """HIGH priority can acquire a reserved slot when unreserved slots are full."""
    mgr = fail_closed_manager
    redis_client = mgr._redis_client

    # Fill the 3 unreserved slots (indices 0-2); slot 3 is reserved
    for i in range(3):
        redis_client.set(f"{mgr._key_prefix}:large:{i}", "other-worker", ex=60)

    # HIGH priority can still acquire slot 3 (the reserved one)
    with mgr.slot("large", priority=Priority.HIGH):
        keys = redis_client.keys(f"{mgr._key_prefix}:large:*")
        assert len(keys) == 4  # 3 filled + 1 acquired


def test_timeout_raises_slot_unavailable(
    fail_closed_manager: SlotManager,
) -> None:
    """When all slots are taken, a timeout raises SlotUnavailable."""
    mgr = fail_closed_manager
    redis_client = mgr._redis_client

    # Fill all 4 slots in "large"
    for i in range(4):
        redis_client.set(f"{mgr._key_prefix}:large:{i}", "other-worker", ex=60)

    start = time.monotonic()
    with pytest.raises(SlotUnavailable) as exc_info:
        with mgr.slot("large", priority=Priority.HIGH, timeout=1):
            pass
    elapsed = time.monotonic() - start

    assert exc_info.value.tier == "large"
    assert exc_info.value.timeout == 1
    # Should not have waited much longer than the 1s timeout
    assert elapsed < 3.0


def test_fail_open_when_redis_unavailable() -> None:
    """With fail_open=True, a disconnected Redis lets the call through."""
    mgr = SlotManager(
        redis_url="redis://localhost:19999/0",  # nothing listening here
        tiers={"large": {"slots": 4, "reserved": 1}},
        priority_timeouts={p: 1 for p in Priority},
        fail_open=True,
    )
    # Should not raise; Redis connection failure is swallowed
    with mgr.slot("large", priority=Priority.HIGH):
        pass  # body executes without a slot — fail-open


def test_fail_closed_when_redis_unavailable() -> None:
    """With fail_open=False, a disconnected Redis raises SlotUnavailable."""
    mgr = SlotManager(
        redis_url="redis://localhost:19999/0",
        tiers={"large": {"slots": 4, "reserved": 1}},
        priority_timeouts={p: 1 for p in Priority},
        fail_open=False,
    )
    with pytest.raises(SlotUnavailable):
        with mgr.slot("large", priority=Priority.HIGH):
            pass


def test_slot_key_has_correct_ttl(manager: SlotManager) -> None:
    """Acquired slot key should have a TTL close to the configured value."""
    redis_client = manager._redis_client

    with manager.slot("large", priority=Priority.HIGH):
        keys = redis_client.keys(f"{manager._key_prefix}:large:*")
        assert len(keys) == 1
        ttl = redis_client.ttl(keys[0])
        # TTL should be within a couple seconds of the configured 60s
        assert 55 <= ttl <= 60


def test_slot_value_is_worker_id(manager: SlotManager) -> None:
    """The Redis key value should be the manager's worker_id."""
    redis_client = manager._redis_client

    with manager.slot("large", priority=Priority.MEDIUM):
        keys = redis_client.keys(f"{manager._key_prefix}:large:*")
        assert len(keys) == 1
        value = redis_client.get(keys[0])
        assert value == manager._worker_id


def test_concurrent_acquisitions_respect_slot_limit(
    fake_redis_server: fakeredis.FakeServer,
) -> None:
    """Under concurrent load, no more than N slots are held at once."""
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={"small": {"slots": 2, "reserved": 0}},
        priority_timeouts={p: 5 for p in Priority},
        ttl=60,
        fail_open=True,
    )
    fake_client = fakeredis.FakeRedis(server=fake_redis_server, decode_responses=True)
    mgr._redis_client = fake_client
    mgr._release_script = None
    mgr._refresh_script = None

    max_observed: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            with mgr.slot("small", priority=Priority.MEDIUM, timeout=5):
                count = len(fake_client.keys(f"{mgr._key_prefix}:small:*"))
                with lock:
                    max_observed.append(count)
                time.sleep(0.05)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Unexpected errors: {errors}"
    # At no point should more than 2 slots be held simultaneously
    assert max(max_observed) <= 2
