"""Tests for async_slot() context manager."""

from __future__ import annotations

import asyncio

import pytest
import fakeredis

from llm_slot_manager import Priority, SlotManager, SlotUnavailable


@pytest.fixture()
def async_manager(fake_redis_server: fakeredis.FakeServer) -> SlotManager:
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={
            "large": {"slots": 4, "reserved": 1},
            "small": {"slots": 2, "reserved": 0},
        },
        priority_timeouts={p: 2 for p in Priority},
        ttl=60,
        fail_open=True,
    )
    fake_client = fakeredis.FakeRedis(server=fake_redis_server, decode_responses=True)
    mgr._redis_client = fake_client
    mgr._release_script = None
    mgr._refresh_script = None
    return mgr


async def test_async_slot_acquired_and_released(async_manager: SlotManager) -> None:
    redis_client = async_manager._redis_client

    async with async_manager.async_slot("large", priority=Priority.HIGH):
        keys = redis_client.keys(f"{async_manager._key_prefix}:large:*")
        assert len(keys) == 1

    keys = redis_client.keys(f"{async_manager._key_prefix}:large:*")
    assert len(keys) == 0


async def test_async_slot_released_on_exception(async_manager: SlotManager) -> None:
    redis_client = async_manager._redis_client

    with pytest.raises(ValueError):
        async with async_manager.async_slot("large", priority=Priority.MEDIUM):
            raise ValueError("body error")

    keys = redis_client.keys(f"{async_manager._key_prefix}:large:*")
    assert len(keys) == 0


async def test_async_slot_unknown_tier_raises(async_manager: SlotManager) -> None:
    with pytest.raises(KeyError, match="no-such-tier"):
        async with async_manager.async_slot("no-such-tier"):
            pass


async def test_async_concurrent_acquisitions(async_manager: SlotManager) -> None:
    """Multiple coroutines can hold slots concurrently up to the limit."""
    redis_client = async_manager._redis_client
    observed_counts: list[int] = []

    async def hold_slot() -> None:
        async with async_manager.async_slot("small", priority=Priority.MEDIUM, timeout=5):
            count = len(redis_client.keys(f"{async_manager._key_prefix}:small:*"))
            observed_counts.append(count)
            await asyncio.sleep(0.05)

    await asyncio.gather(*[hold_slot() for _ in range(4)])

    # At no point should more than 2 slots (the tier limit) be occupied
    assert max(observed_counts) <= 2


async def test_async_fail_open_on_redis_unavailable() -> None:
    mgr = SlotManager(
        redis_url="redis://localhost:19998/0",
        tiers={"large": {"slots": 4}},
        priority_timeouts={p: 1 for p in Priority},
        fail_open=True,
    )
    # Should not raise
    async with mgr.async_slot("large"):
        pass
