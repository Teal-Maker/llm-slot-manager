"""Priority-aware distributed semaphore for LLM inference slots.

Each tier (e.g. "large", "small") has a fixed number of semaphore slots backed
by Redis SET NX keys with TTL.  Workers acquire a slot before making an LLM
HTTP call and release it immediately after.

Priority system
---------------
CRITICAL / HIGH / MEDIUM tasks can access all slots, including reserved ones.
LOW tasks can only access unreserved slots (indices 0 .. total-reserved-1).
This guarantees that at least ``reserved`` slots remain available for
higher-priority work even under full load from background tasks.

Fail-open behaviour
-------------------
When Redis is unavailable (connection error, timeout) and ``fail_open=True``
(the default), acquisition returns immediately without blocking.  The caller
proceeds without a slot, trading back-pressure for availability.  Set
``fail_open=False`` to raise :class:`~llm_slot_manager.exceptions.SlotUnavailable`
instead.

TTL refresh
-----------
Each acquired slot key is given a TTL (default 660 s) to prevent stuck keys if
a worker crashes.  A lightweight daemon thread refreshes the TTL every
``ttl_refresh_interval`` seconds so that long-running requests do not lose their
slot.  Refreshes are capped at ``ttl_refresh_max_count`` to prevent zombie slots
from persisting indefinitely.

Prometheus metrics
------------------
Pass ``enable_metrics=True`` to the constructor to enable Prometheus counters,
gauges, and histograms.  The ``prometheus-client`` package must be installed
(available via the ``metrics`` extra).  If the package is absent, a
:class:`ImportError` is raised at construction time when metrics are requested.

Example::

    from llm_slot_manager import SlotManager, Priority

    manager = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={
            "large": {"slots": 4, "reserved": 1},
            "small": {"slots": 6, "reserved": 1},
        },
        priority_timeouts={
            Priority.CRITICAL: 120,
            Priority.HIGH: 60,
            Priority.MEDIUM: 30,
            Priority.LOW: 10,
        },
    )

    with manager.slot("large", priority=Priority.HIGH):
        response = requests.post(endpoint, json=payload)

    async with manager.async_slot("small", priority=Priority.LOW):
        response = await client.post(endpoint, json=payload)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Generator, AsyncGenerator, Optional

import redis

from llm_slot_manager.exceptions import SlotUnavailable
from llm_slot_manager.priority import Priority

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lua scripts
# ---------------------------------------------------------------------------

_RELEASE_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""

_REFRESH_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("EXPIRE", KEYS[1], ARGV[2])
else
    return 0
end
"""

# ---------------------------------------------------------------------------
# TierConfig — typed view of a single tier's configuration
# ---------------------------------------------------------------------------

class TierConfig:
    """Validated configuration for a single slot tier."""

    __slots__ = ("name", "slots", "reserved")

    def __init__(self, name: str, slots: int, reserved: int) -> None:
        if ":" in name:
            raise ValueError(
                f"Tier name '{name}' must not contain ':' (reserved for key formatting)"
            )
        if slots < 1:
            raise ValueError(f"Tier '{name}': slots must be >= 1, got {slots}")
        if reserved < 0:
            raise ValueError(f"Tier '{name}': reserved must be >= 0, got {reserved}")
        if reserved >= slots:
            raise ValueError(
                f"Tier '{name}': reserved ({reserved}) must be < slots ({slots})"
            )
        self.name = name
        self.slots = slots
        self.reserved = reserved


# ---------------------------------------------------------------------------
# _TTLRefresher — daemon thread for long-running slot TTL extension
# ---------------------------------------------------------------------------

class _TTLRefresher:
    """Background daemon thread that refreshes a slot's TTL during long requests.

    The refresher is stopped (via :meth:`stop`) when the slot is released.
    It runs as a daemon thread so it will not prevent interpreter shutdown.
    Refreshes are capped at *max_count* to prevent zombie slots.
    """

    def __init__(
        self,
        key: str,
        worker_id: str,
        ttl: int,
        refresh_interval: int,
        max_count: int,
        refresh_script: Any | None,
        get_redis_fn: Any,
        metrics_refresh_counter: Any | None = None,
    ) -> None:
        self._key = key
        self._worker_id = worker_id
        self._ttl = ttl
        self._refresh_interval = refresh_interval
        self._max_count = max_count
        self._refresh_script = refresh_script
        self._get_redis = get_redis_fn
        self._metrics_refresh_counter = metrics_refresh_counter

        self._stop_event = threading.Event()
        self._refresh_count = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background refresh thread."""
        self._thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name=f"ttl-refresh-{self._key}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the refresh thread to stop.  Does not block."""
        self._stop_event.set()

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._refresh_interval):
            if self._refresh_count >= self._max_count:
                logger.warning(
                    "TTL refresh cap reached for %s (%d refreshes, ~%ds total extension)",
                    self._key,
                    self._max_count,
                    self._max_count * self._refresh_interval,
                )
                return

            r = self._get_redis()
            if r is None:
                return

            try:
                if self._refresh_script is not None:
                    refreshed = self._refresh_script(
                        keys=[self._key],
                        args=[self._worker_id, self._ttl],
                    )
                else:
                    # Fallback if script was not registered (connection was replaced)
                    if r.get(self._key) == self._worker_id:
                        r.expire(self._key, self._ttl)
                        refreshed = 1
                    else:
                        refreshed = 0

                self._refresh_count += 1

                if refreshed:
                    if self._metrics_refresh_counter is not None:
                        # Extract tier from key pattern: {prefix}:{tier}:{idx}
                        parts = self._key.split(":")
                        tier = parts[-2] if len(parts) >= 3 else "unknown"
                        self._metrics_refresh_counter.labels(tier=tier).inc()
                    logger.debug(
                        "Refreshed TTL for %s (refresh %d/%d)",
                        self._key,
                        self._refresh_count,
                        self._max_count,
                    )
                else:
                    logger.warning(
                        "TTL refresh failed for %s — slot no longer owned", self._key
                    )
                    return
            except redis.RedisError as exc:
                logger.warning("Redis error refreshing TTL for %s: %s", self._key, exc)


# ---------------------------------------------------------------------------
# SlotManager — public API
# ---------------------------------------------------------------------------

class SlotManager:
    """Priority-aware distributed semaphore for LLM inference slots.

    Parameters
    ----------
    redis_url:
        Redis connection URL, e.g. ``"redis://localhost:6379/0"``.
    tiers:
        Mapping of tier name to configuration dict with keys:

        - ``"slots"`` (int, required): total number of slots for this tier.
        - ``"reserved"`` (int, optional, default 0): slots reserved for
          MEDIUM+ priority tasks; LOW priority cannot use these slots.
    priority_timeouts:
        Per-priority maximum wait time in seconds.  Defaults to
        ``{CRITICAL: 120, HIGH: 60, MEDIUM: 30, LOW: 10}`` if not provided.
    ttl:
        Slot key TTL in seconds.  A slot key that outlives this TTL (e.g. due
        to a crashed worker) will be automatically expired by Redis.
        Default: 660.
    fail_open:
        When ``True`` (default), Redis errors cause acquisition to return
        immediately without a slot rather than raising.  When ``False``,
        a :class:`~llm_slot_manager.exceptions.SlotUnavailable` is raised.
    enable_metrics:
        When ``True``, register Prometheus metrics for this instance.
        Requires ``prometheus-client`` to be installed.  Default: ``False``.
    key_prefix:
        Redis key prefix.  Keys follow the pattern
        ``{key_prefix}:{tier}:{index}``.  Default: ``"llm:slot"``.
    ttl_refresh_interval:
        How often (seconds) the TTL refresher daemon wakes to extend the
        slot key's expiry.  Default: 300.
    ttl_refresh_max_count:
        Maximum number of TTL refreshes per slot acquisition.  Prevents
        zombie slots from being extended indefinitely.  Default: 3.
    """

    _DEFAULT_PRIORITY_TIMEOUTS: dict[Priority, int] = {
        Priority.CRITICAL: 120,
        Priority.HIGH: 60,
        Priority.MEDIUM: 30,
        Priority.LOW: 10,
    }

    def __init__(
        self,
        redis_url: str,
        tiers: dict[str, dict[str, int]],
        priority_timeouts: dict[Priority, int] | None = None,
        ttl: int = 660,
        fail_open: bool = True,
        enable_metrics: bool = False,
        key_prefix: str = "llm:slot",
        ttl_refresh_interval: int = 300,
        ttl_refresh_max_count: int = 3,
    ) -> None:
        if not tiers:
            raise ValueError("At least one tier must be configured")
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        if ttl_refresh_interval <= 0:
            raise ValueError(f"ttl_refresh_interval must be > 0, got {ttl_refresh_interval}")
        if ttl_refresh_max_count < 0:
            raise ValueError(f"ttl_refresh_max_count must be >= 0, got {ttl_refresh_max_count}")

        self._redis_url = redis_url
        self._ttl = ttl
        self._fail_open = fail_open
        self._key_prefix = key_prefix
        self._ttl_refresh_interval = ttl_refresh_interval
        self._ttl_refresh_max_count = ttl_refresh_max_count

        self._priority_timeouts: dict[Priority, int] = (
            priority_timeouts if priority_timeouts is not None
            else dict(self._DEFAULT_PRIORITY_TIMEOUTS)
        )

        # Parse and validate tier configs
        self._tiers: dict[str, TierConfig] = {}
        for name, cfg in tiers.items():
            if "slots" not in cfg:
                raise ValueError(f"Tier '{name}' is missing required key 'slots'")
            self._tiers[name] = TierConfig(
                name=name,
                slots=cfg["slots"],
                reserved=cfg.get("reserved", 0),
            )

        # Unique identity for this manager instance (per-process, per-instance)
        self._worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

        # Lazy Redis state — populated on first use
        self._redis_client: Optional[redis.Redis] = None  # type: ignore[type-arg]
        self._release_script: Any = None
        self._refresh_script: Any = None
        self._redis_lock = threading.Lock()

        # Prometheus metrics (per-instance, created lazily below)
        self._metrics_acquired: Any = None
        self._metrics_timeout: Any = None
        self._metrics_active: Any = None
        self._metrics_wait: Any = None
        self._metrics_reserved_skip: Any = None
        self._metrics_ttl_refresh: Any = None

        if enable_metrics:
            self._init_metrics()

        logger.debug(
            "SlotManager created (worker_id=%s, tiers=%s, ttl=%ds, fail_open=%s)",
            self._worker_id,
            {n: {"slots": c.slots, "reserved": c.reserved} for n, c in self._tiers.items()},
            self._ttl,
            self._fail_open,
        )

    # ------------------------------------------------------------------
    # Public context managers
    # ------------------------------------------------------------------

    @contextmanager
    def slot(
        self,
        tier: str,
        priority: Priority = Priority.MEDIUM,
        timeout: int | None = None,
    ) -> Generator[None, None, None]:
        """Synchronous context manager for acquiring a slot.

        Parameters
        ----------
        tier:
            The tier name to acquire a slot from.
        priority:
            Task priority.  Affects both the wait timeout and whether reserved
            slots are accessible.  Defaults to :attr:`~Priority.MEDIUM`.
        timeout:
            Override the priority-derived wait timeout (seconds).

        Raises
        ------
        SlotUnavailable
            If all accessible slots are occupied and the timeout expires
            (only when ``fail_open=False``).
        KeyError
            If *tier* is not a configured tier name.

        Example::

            with manager.slot("large", priority=Priority.HIGH):
                response = requests.post(endpoint, json=payload)
        """
        self._validate_tier(tier)
        key = self._acquire_slot(tier, priority, timeout)
        refresher: Optional[_TTLRefresher] = None
        if key is not None:
            refresher = self._make_refresher(key)
            refresher.start()
        try:
            yield
        finally:
            self._release_slot(key, refresher)

    @asynccontextmanager
    async def async_slot(
        self,
        tier: str,
        priority: Priority = Priority.MEDIUM,
        timeout: int | None = None,
    ) -> AsyncGenerator[None, None]:
        """Async context manager for acquiring a slot.

        Slot acquisition runs in a thread executor so the event loop is not
        blocked during contention waits (which can be up to 120 s for
        CRITICAL priority).

        Parameters
        ----------
        tier:
            The tier name to acquire a slot from.
        priority:
            Task priority.  Defaults to :attr:`~Priority.MEDIUM`.
        timeout:
            Override the priority-derived wait timeout (seconds).

        Raises
        ------
        SlotUnavailable
            If all accessible slots are occupied and the timeout expires
            (only when ``fail_open=False``).
        KeyError
            If *tier* is not a configured tier name.

        Example::

            async with manager.async_slot("small", priority=Priority.LOW):
                response = await client.post(endpoint, json=payload)
        """
        self._validate_tier(tier)
        key = await asyncio.to_thread(self._acquire_slot, tier, priority, timeout)
        refresher: Optional[_TTLRefresher] = None
        if key is not None:
            refresher = self._make_refresher(key)
            refresher.start()
        try:
            yield
        finally:
            self._release_slot(key, refresher)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_occupancy(self, tier: str) -> tuple[int, int]:
        """Return ``(occupied, total)`` slot counts for a tier.

        Counts occupied slots by scanning Redis keys for all slot indices.
        Returns ``(0, total)`` when Redis is unavailable.

        Parameters
        ----------
        tier:
            The tier name to inspect.

        Raises
        ------
        KeyError
            If *tier* is not a configured tier name.

        Example::

            occupied, total = manager.get_occupancy("large")
            print(f"{occupied}/{total} slots in use")
        """
        self._validate_tier(tier)
        cfg = self._tiers[tier]
        total = cfg.slots

        r = self._get_redis()
        if r is None:
            return (0, total)

        occupied = 0
        try:
            for idx in range(total):
                key = f"{self._key_prefix}:{tier}:{idx}"
                if r.exists(key):
                    occupied += 1
        except redis.RedisError as exc:
            logger.warning("Redis error checking occupancy for tier '%s': %s", tier, exc)
            return (0, total)

        return (occupied, total)

    def get_utilization(self, tier: str) -> float:
        """Return slot utilization as a fraction in ``[0.0, 1.0]`` for a tier.

        Returns ``0.0`` when Redis is unavailable or the tier has no slots.

        Parameters
        ----------
        tier:
            The tier name to inspect.

        Raises
        ------
        KeyError
            If *tier* is not a configured tier name.

        Example::

            util = manager.get_utilization("large")
            if util > 0.9:
                logger.warning("Large tier near capacity (%.0f%%)", util * 100)
        """
        occupied, total = self.get_occupancy(tier)
        if total == 0:
            return 0.0
        return occupied / total

    # ------------------------------------------------------------------
    # Internal: Redis connection
    # ------------------------------------------------------------------

    def _get_redis(self) -> Optional[redis.Redis]:  # type: ignore[type-arg]
        """Return a connected Redis client, lazily creating it on first call.

        Thread-safe.  Returns ``None`` on connection failure (fail-open).
        """
        # Fast path — already connected
        if self._redis_client is not None:
            return self._redis_client

        with self._redis_lock:
            # Double-checked locking
            if self._redis_client is not None:
                return self._redis_client

            try:
                client: redis.Redis = redis.Redis.from_url(  # type: ignore[type-arg]
                    self._redis_url,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    retry_on_timeout=False,
                )
                client.ping()
                self._release_script = client.register_script(_RELEASE_LUA)
                self._refresh_script = client.register_script(_REFRESH_LUA)
                self._redis_client = client
                logger.info(
                    "SlotManager connected to Redis (worker_id=%s, tiers=%s, ttl=%ds)",
                    self._worker_id,
                    list(self._tiers),
                    self._ttl,
                )
            except (redis.RedisError, ConnectionError, OSError) as exc:
                logger.warning(
                    "SlotManager Redis connection failed (fail-open=%s): %s",
                    self._fail_open,
                    exc,
                )
                return None

        return self._redis_client

    # ------------------------------------------------------------------
    # Internal: slot acquire / release
    # ------------------------------------------------------------------

    def _acquire_slot(
        self,
        tier: str,
        priority: Priority,
        timeout: int | None,
    ) -> Optional[str]:
        """Try to acquire a slot for *tier* with *priority*.

        Returns the Redis key of the acquired slot, or ``None`` if the
        manager is operating fail-open and Redis is unavailable.

        Raises :class:`SlotUnavailable` if the timeout expires (fail-open
        applies only to Redis connection failures, not to timeout exhaustion).
        """
        r = self._get_redis()
        if r is None:
            if self._fail_open:
                return None
            cfg = self._tiers[tier]
            effective_timeout = timeout if timeout is not None else self._priority_timeouts.get(priority, 30)
            raise SlotUnavailable(tier, effective_timeout, priority, cfg.slots)

        cfg = self._tiers[tier]
        effective_timeout = (
            timeout if timeout is not None
            else self._priority_timeouts.get(priority, 30)
        )

        max_slots = cfg.slots
        reserved = cfg.reserved

        # LOW priority cannot consume reserved slots
        if priority > Priority.MEDIUM:
            accessible_slots = max_slots - reserved
        else:
            accessible_slots = max_slots

        if accessible_slots <= 0:
            # All slots are reserved — LOW is locked out entirely
            if self._metrics_reserved_skip is not None:
                self._metrics_reserved_skip.labels(tier=tier).inc()
            if self._metrics_timeout is not None:
                self._metrics_timeout.labels(tier=tier, priority=priority.name).inc()
            raise SlotUnavailable(tier, effective_timeout, priority, max_slots)

        deadline = time.monotonic() + effective_timeout
        wait_start = time.monotonic()
        slot_indices = list(range(accessible_slots))
        reserved_skip_counted = False

        while True:
            random.shuffle(slot_indices)

            for idx in slot_indices:
                key = f"{self._key_prefix}:{tier}:{idx}"
                try:
                    acquired = r.set(key, self._worker_id, nx=True, ex=self._ttl)
                except redis.RedisError as exc:
                    logger.warning(
                        "Redis error acquiring slot %s (fail-open=%s): %s",
                        key,
                        self._fail_open,
                        exc,
                    )
                    if self._fail_open:
                        return None
                    raise SlotUnavailable(tier, effective_timeout, priority, max_slots) from exc

                if acquired:
                    wait_time = time.monotonic() - wait_start
                    if self._metrics_acquired is not None:
                        self._metrics_acquired.labels(tier=tier, priority=priority.name).inc()
                    if self._metrics_active is not None:
                        self._metrics_active.labels(tier=tier).inc()
                    if self._metrics_wait is not None:
                        self._metrics_wait.labels(tier=tier, priority=priority.name).observe(
                            wait_time
                        )
                    if wait_time > 1.0:
                        logger.info(
                            "Acquired slot %s:%d after %.1fs wait (priority=%s)",
                            tier,
                            idx,
                            wait_time,
                            priority.name,
                        )
                    else:
                        logger.debug(
                            "Acquired slot %s:%d (priority=%s)", tier, idx, priority.name
                        )
                    return key

            # Track reserved-slot skips for LOW priority (once per acquire call)
            if not reserved_skip_counted and priority > Priority.MEDIUM and reserved > 0:
                reserved_skip_counted = True
                if self._metrics_reserved_skip is not None:
                    self._metrics_reserved_skip.labels(tier=tier).inc()

            # All accessible slots occupied — check deadline
            if time.monotonic() >= deadline:
                if self._metrics_timeout is not None:
                    self._metrics_timeout.labels(tier=tier, priority=priority.name).inc()
                if self._metrics_wait is not None:
                    self._metrics_wait.labels(tier=tier, priority=priority.name).observe(
                        time.monotonic() - wait_start
                    )
                raise SlotUnavailable(tier, effective_timeout, priority, max_slots)

            # Back off with jitter before retrying
            sleep_time = 0.5 + random.uniform(0, 0.3)
            remaining = deadline - time.monotonic()
            time.sleep(min(sleep_time, max(remaining, 0.01)))

    def _release_slot(
        self,
        key: Optional[str],
        refresher: Optional[_TTLRefresher],
    ) -> None:
        """Release a slot key (atomic check-and-delete owned by this worker)."""
        if refresher is not None:
            refresher.stop()

        if key is None:
            return

        r = self._get_redis()
        if r is None:
            return

        try:
            if self._release_script is not None:
                released = self._release_script(keys=[key], args=[self._worker_id])
            else:
                # Fallback: non-atomic, only used if script registration failed
                if r.get(key) == self._worker_id:
                    r.delete(key)
                    released = 1
                else:
                    released = 0

            if released:
                # Extract tier from key: {prefix}:{tier}:{idx}
                parts = key.split(":")
                tier = parts[-2] if len(parts) >= 3 else "unknown"
                if self._metrics_active is not None:
                    self._metrics_active.labels(tier=tier).dec()
                logger.debug("Released slot %s", key)
            else:
                logger.debug("Slot %s already released or owned by another worker", key)
        except redis.RedisError as exc:
            logger.warning("Redis error releasing slot %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _validate_tier(self, tier: str) -> None:
        if tier not in self._tiers:
            raise KeyError(
                f"Unknown tier '{tier}'. Configured tiers: {list(self._tiers)}"
            )

    def _make_refresher(self, key: str) -> _TTLRefresher:
        return _TTLRefresher(
            key=key,
            worker_id=self._worker_id,
            ttl=self._ttl,
            refresh_interval=self._ttl_refresh_interval,
            max_count=self._ttl_refresh_max_count,
            refresh_script=self._refresh_script,
            get_redis_fn=self._get_redis,
            metrics_refresh_counter=self._metrics_ttl_refresh,
        )

    def _init_metrics(self) -> None:
        """Create per-instance Prometheus metrics.

        Uses a unique registry suffix derived from the worker_id so that
        multiple SlotManager instances in the same process do not collide
        on metric names.  If ``prometheus-client`` is not installed, raises
        :class:`ImportError` immediately.
        """
        try:
            from prometheus_client import Counter, Gauge, Histogram  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "enable_metrics=True requires 'prometheus-client'. "
                "Install it with: pip install 'llm-slot-manager[metrics]'"
            ) from exc

        # Suffix with worker_id to avoid duplicate-metric errors when
        # multiple SlotManager instances coexist in the same process.
        suffix = self._worker_id.replace("-", "_")

        self._metrics_acquired = Counter(
            f"llm_slot_acquired_total_{suffix}",
            "Total slot acquisitions",
            ["tier", "priority"],
        )
        self._metrics_timeout = Counter(
            f"llm_slot_timeout_total_{suffix}",
            "Total slot acquisition timeouts",
            ["tier", "priority"],
        )
        self._metrics_active = Gauge(
            f"llm_slot_active_{suffix}",
            "Currently held slots",
            ["tier"],
        )
        self._metrics_wait = Histogram(
            f"llm_slot_wait_seconds_{suffix}",
            "Time spent waiting for a slot",
            ["tier", "priority"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        self._metrics_reserved_skip = Counter(
            f"llm_slot_reserved_skip_total_{suffix}",
            "Times LOW priority was blocked by a reserved slot",
            ["tier"],
        )
        self._metrics_ttl_refresh = Counter(
            f"llm_slot_ttl_refresh_total_{suffix}",
            "Total TTL refreshes for long-running requests",
            ["tier"],
        )
        logger.debug("Prometheus metrics enabled for SlotManager (suffix=%s)", suffix)
