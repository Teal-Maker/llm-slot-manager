# llm-slot-manager

Priority-aware distributed semaphore for LLM inference endpoints, backed by Redis.

## Why use this?

If you run multiple LLM inference servers (llama.cpp, vLLM, Ollama, etc.) with a fixed number of concurrent slots, you need a way to prevent workers from overwhelming them with unbounded requests. This library gives you a Redis-backed semaphore that understands priorities: critical tasks always get through, while best-effort work waits or gets rejected when capacity is full.

**Use this when you have:**
- Multiple workers or services sharing a pool of LLM inference endpoints
- Different task priorities (e.g. user-facing requests vs background batch jobs)
- A need to reserve capacity for high-priority work while letting low-priority tasks fill remaining slots
- Redis already in your stack

## Installation

```bash
pip install llm-slot-manager
# With Prometheus metrics support:
pip install "llm-slot-manager[metrics]"
```

## Quick Start

```python
from llm_slot_manager import SlotManager, Priority, SlotUnavailable

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

# Synchronous
with manager.slot("large", priority=Priority.HIGH):
    response = requests.post(endpoint, json=payload)

# Asynchronous
async with manager.async_slot("small", priority=Priority.LOW):
    response = await client.post(endpoint, json=payload)

# Observability
occupied, total = manager.get_occupancy("large")
utilization = manager.get_utilization("small")
```

## Priority Levels

| Priority | Value | Reserved Slots |
|----------|-------|----------------|
| CRITICAL | 1     | accessible     |
| HIGH     | 2     | accessible     |
| MEDIUM   | 3     | accessible     |
| LOW      | 4     | blocked        |

Lower numeric value = higher priority. LOW priority tasks cannot consume
reserved slots, ensuring headroom for latency-sensitive work.

## License

Apache-2.0
