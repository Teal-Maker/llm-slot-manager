"""Tests for SlotUnavailable exception."""

import pytest

from llm_slot_manager import Priority, SlotUnavailable


def test_slot_unavailable_basic() -> None:
    exc = SlotUnavailable("large", 30, Priority.HIGH)
    assert exc.tier == "large"
    assert exc.timeout == 30
    assert exc.priority == Priority.HIGH
    assert exc.total_slots is None
    assert "large" in str(exc)
    assert "HIGH" in str(exc)
    assert "30" in str(exc)


def test_slot_unavailable_with_total_slots() -> None:
    exc = SlotUnavailable("small", 10, Priority.LOW, total_slots=6)
    assert exc.total_slots == 6
    assert "6" in str(exc)


def test_slot_unavailable_default_priority() -> None:
    exc = SlotUnavailable("large", 30)
    assert exc.priority == Priority.LOW


def test_slot_unavailable_is_exception() -> None:
    exc = SlotUnavailable("large", 30)
    assert isinstance(exc, Exception)


def test_slot_unavailable_can_be_raised_and_caught() -> None:
    with pytest.raises(SlotUnavailable) as exc_info:
        raise SlotUnavailable("xlarge", 60, Priority.CRITICAL, total_slots=2)

    assert exc_info.value.tier == "xlarge"
    assert exc_info.value.priority == Priority.CRITICAL
