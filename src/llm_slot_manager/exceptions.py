"""Exceptions raised by llm-slot-manager."""

from llm_slot_manager.priority import Priority


class SlotUnavailable(Exception):
    """Raised when no slot could be acquired within the configured timeout.

    Attributes:
        tier: The tier name that was requested.
        timeout: The timeout in seconds that was exhausted.
        priority: The priority level that was used for the acquisition attempt.
        total_slots: Total slots configured for the tier (if known).

    Example::

        from llm_slot_manager import SlotManager, Priority, SlotUnavailable

        try:
            with manager.slot("large", priority=Priority.LOW):
                ...
        except SlotUnavailable as exc:
            logger.warning(
                "No slot available for tier %s after %ds (priority=%s)",
                exc.tier, exc.timeout, exc.priority.name,
            )
    """

    def __init__(
        self,
        tier: str,
        timeout: int,
        priority: Priority = Priority.LOW,
        total_slots: int | None = None,
    ) -> None:
        self.tier = tier
        self.timeout = timeout
        self.priority = priority
        self.total_slots = total_slots

        slots_info = f" — all {total_slots} slots occupied" if total_slots is not None else ""
        super().__init__(
            f"Could not acquire slot for tier '{tier}' "
            f"(priority={priority.name}) within {timeout}s{slots_info}"
        )
