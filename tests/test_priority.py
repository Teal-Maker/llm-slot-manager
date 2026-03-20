"""Tests for the Priority enum."""

import pytest

from llm_slot_manager import Priority


def test_priority_values() -> None:
    assert Priority.CRITICAL == 1
    assert Priority.HIGH == 2
    assert Priority.MEDIUM == 3
    assert Priority.LOW == 4


def test_priority_ordering() -> None:
    """Lower numeric value means higher priority."""
    assert Priority.CRITICAL < Priority.HIGH
    assert Priority.HIGH < Priority.MEDIUM
    assert Priority.MEDIUM < Priority.LOW


def test_priority_is_int_enum() -> None:
    from enum import IntEnum
    assert issubclass(Priority, IntEnum)


def test_priority_comparison_with_int() -> None:
    assert Priority.LOW >= Priority.LOW
    assert Priority.MEDIUM < Priority.LOW
    assert Priority.HIGH < Priority.MEDIUM


def test_priority_names() -> None:
    names = [p.name for p in Priority]
    assert names == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
