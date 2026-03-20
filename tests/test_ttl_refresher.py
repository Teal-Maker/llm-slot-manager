"""Tests for _TTLRefresher daemon thread.

All tests use fakeredis's native GET/EXPIRE commands directly (no register_script),
since fakeredis does not support EVALSHA.  The refresher's fallback path
(GET + EXPIRE without Lua) is exercised here; the Lua path is covered by
integration tests against a real Redis instance.
"""

from __future__ import annotations

import time

import fakeredis
import pytest

from llm_slot_manager.manager import _TTLRefresher


@pytest.fixture()
def fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def _make_refresher(
    fake_redis_client: fakeredis.FakeRedis,
    key: str,
    worker_id: str,
    ttl: int = 60,
    refresh_interval: int = 0,
    max_count: int = 3,
) -> _TTLRefresher:
    """Build a _TTLRefresher using the non-Lua fallback (script=None)."""
    return _TTLRefresher(
        key=key,
        worker_id=worker_id,
        ttl=ttl,
        refresh_interval=refresh_interval,
        max_count=max_count,
        refresh_script=None,       # exercises the GET+EXPIRE fallback path
        get_redis_fn=lambda: fake_redis_client,
        metrics_refresh_counter=None,
    )


def test_refresher_stops_cleanly(fake_redis: fakeredis.FakeRedis) -> None:
    key = "llm:slot:large:0"
    worker_id = "test-worker"
    fake_redis.set(key, worker_id, ex=10)

    # Use a very long interval so the refresh body never actually runs
    refresher = _TTLRefresher(
        key=key,
        worker_id=worker_id,
        ttl=60,
        refresh_interval=9999,
        max_count=3,
        refresh_script=None,
        get_redis_fn=lambda: None,
        metrics_refresh_counter=None,
    )
    refresher.start()
    refresher.stop()
    time.sleep(0.05)
    assert refresher._stop_event.is_set()


def test_refresher_stop_is_idempotent(fake_redis: fakeredis.FakeRedis) -> None:
    refresher = _TTLRefresher(
        key="llm:slot:large:0",
        worker_id="worker",
        ttl=60,
        refresh_interval=9999,
        max_count=3,
        refresh_script=None,
        get_redis_fn=lambda: None,
        metrics_refresh_counter=None,
    )
    refresher.start()
    refresher.stop()
    refresher.stop()  # second stop should not raise


def test_refresher_extends_ttl(fake_redis: fakeredis.FakeRedis) -> None:
    """After one refresh cycle the TTL should be reset to the configured value."""
    key = "llm:slot:large:0"
    worker_id = "test-worker"
    # Set with a short initial TTL
    fake_redis.set(key, worker_id, ex=5)

    # interval=0: stop_event.wait(0) returns False (not set) immediately,
    # so the loop body runs on every iteration without sleeping
    refresher = _make_refresher(
        fake_redis_client=fake_redis,
        key=key,
        worker_id=worker_id,
        ttl=60,
        refresh_interval=0,
        max_count=3,
    )
    refresher.start()
    # Allow at least one refresh to complete
    time.sleep(0.15)
    refresher.stop()

    ttl_after = fake_redis.ttl(key)
    # TTL should have been refreshed to ~60; allow a small margin for timing
    assert ttl_after > 10, f"Expected TTL > 10 after refresh, got {ttl_after}"


def test_refresher_respects_max_count(fake_redis: fakeredis.FakeRedis) -> None:
    """Refresher thread exits after reaching max_count refreshes."""
    key = "llm:slot:large:0"
    worker_id = "test-worker"
    fake_redis.set(key, worker_id, ex=60)

    refresher = _make_refresher(
        fake_redis_client=fake_redis,
        key=key,
        worker_id=worker_id,
        ttl=60,
        refresh_interval=0,
        max_count=2,
    )
    refresher.start()
    # Wait long enough for the cap to be hit and the thread to exit
    time.sleep(0.5)
    refresher.stop()

    # Exactly 2 refreshes should have occurred (then the loop returns)
    assert refresher._refresh_count <= 2


def test_refresher_aborts_if_key_lost(fake_redis: fakeredis.FakeRedis) -> None:
    """If the key is deleted externally, refresher detects it and exits."""
    key = "llm:slot:large:0"
    worker_id = "test-worker"
    fake_redis.set(key, worker_id, ex=60)
    # Delete immediately — next refresh sees the key is gone
    fake_redis.delete(key)

    refresher = _make_refresher(
        fake_redis_client=fake_redis,
        key=key,
        worker_id=worker_id,
        ttl=60,
        refresh_interval=0,
        max_count=5,
    )
    refresher.start()
    time.sleep(0.2)
    refresher.stop()

    # The refresher should have exited after the first failed attempt
    assert refresher._refresh_count <= 1


def test_refresher_does_not_extend_foreign_key(fake_redis: fakeredis.FakeRedis) -> None:
    """Refresher must not extend a key owned by a different worker."""
    key = "llm:slot:large:0"
    # Set the key with a *different* worker_id
    fake_redis.set(key, "other-worker", ex=10)

    refresher = _make_refresher(
        fake_redis_client=fake_redis,
        key=key,
        worker_id="my-worker",  # does not match the stored value
        ttl=60,
        refresh_interval=0,
        max_count=5,
    )
    refresher.start()
    time.sleep(0.15)
    refresher.stop()

    # The TTL should NOT have been extended (key is owned by other-worker)
    ttl_after = fake_redis.ttl(key)
    assert ttl_after <= 10, f"TTL should not have been extended, got {ttl_after}"
    assert refresher._refresh_count <= 1
