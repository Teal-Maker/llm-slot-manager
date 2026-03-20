"""Shared pytest fixtures for llm-slot-manager tests.

Note: fakeredis does not support EVALSHA (the mechanism used by redis-py's
register_script).  All fixtures therefore set the Lua script references to None
so that SlotManager falls back to its non-atomic GET+DEL / GET+EXPIRE paths,
which are semantically equivalent for single-process tests.
"""

from __future__ import annotations

import pytest
import fakeredis

from llm_slot_manager import Priority, SlotManager


SIMPLE_TIERS = {
    "large": {"slots": 4, "reserved": 1},
    "small": {"slots": 2, "reserved": 0},
}

DEFAULT_TIMEOUTS = {
    Priority.CRITICAL: 5,
    Priority.HIGH: 3,
    Priority.MEDIUM: 2,
    Priority.LOW: 1,
}


def _make_manager(
    server: fakeredis.FakeServer,
    fail_open: bool = True,
) -> SlotManager:
    """Wire a SlotManager to an in-process fakeredis instance.

    Scripts are set to None so the manager uses its non-Lua fallback paths,
    which are fully correct for single-process unit tests.
    """
    mgr = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers=SIMPLE_TIERS,
        priority_timeouts=DEFAULT_TIMEOUTS,
        ttl=60,
        fail_open=fail_open,
        enable_metrics=False,
    )
    fake_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    mgr._redis_client = fake_client
    # Leave Lua scripts as None — manager falls back to GET+DEL / GET+EXPIRE
    mgr._release_script = None
    mgr._refresh_script = None
    return mgr


@pytest.fixture()
def fake_redis_server() -> fakeredis.FakeServer:
    """A shared fakeredis server instance (in-process, no network)."""
    return fakeredis.FakeServer()


@pytest.fixture()
def manager(fake_redis_server: fakeredis.FakeServer) -> SlotManager:
    """A SlotManager wired to an in-process fakeredis instance."""
    return _make_manager(fake_redis_server, fail_open=True)


@pytest.fixture()
def fail_closed_manager(fake_redis_server: fakeredis.FakeServer) -> SlotManager:
    """A SlotManager with fail_open=False for testing timeout behaviour."""
    return _make_manager(fake_redis_server, fail_open=False)
