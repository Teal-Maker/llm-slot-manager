"""llm-slot-manager — Priority-aware distributed semaphore for LLM inference endpoints.

Quick start::

    from llm_slot_manager import SlotManager, Priority, SlotUnavailable

    manager = SlotManager(
        redis_url="redis://localhost:6379/0",
        tiers={
            "large": {"slots": 4, "reserved": 1},
            "small": {"slots": 6, "reserved": 1},
        },
    )

    # Synchronous
    with manager.slot("large", priority=Priority.HIGH):
        response = requests.post(endpoint, json=payload)

    # Asynchronous
    async with manager.async_slot("small", priority=Priority.LOW):
        response = await client.post(endpoint, json=payload)

    # Observability
    occupied, total = manager.get_occupancy("large")
    utilization = manager.get_utilization("small")
"""

from llm_slot_manager.exceptions import SlotUnavailable
from llm_slot_manager.manager import SlotManager
from llm_slot_manager.priority import Priority

__all__ = [
    "SlotManager",
    "Priority",
    "SlotUnavailable",
]

__version__ = "0.1.0"
