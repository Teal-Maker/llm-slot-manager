"""Priority levels for LLM slot acquisition."""

from enum import IntEnum


class Priority(IntEnum):
    """Task priority levels for slot acquisition.

    Lower numeric value = higher priority.

    CRITICAL, HIGH, and MEDIUM can access all slots including reserved ones.
    LOW can only access unreserved slots (slots 0..total-reserved-1).

    Examples::

        from llm_slot_manager import Priority

        # High-priority interactive request
        with manager.slot("large", priority=Priority.HIGH):
            response = requests.post(endpoint, json=payload)

        # Background best-effort task
        with manager.slot("small", priority=Priority.LOW):
            response = requests.post(endpoint, json=payload)
    """

    CRITICAL = 1  # Interactive or user-initiated requests requiring immediate response
    HIGH = 2      # Specialized pipeline tasks with latency requirements
    MEDIUM = 3    # Standard pipeline tasks (default for most workers)
    LOW = 4       # Best-effort background tasks; cannot consume reserved slots
