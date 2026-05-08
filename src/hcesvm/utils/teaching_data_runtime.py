"""Runtime helpers for teaching-data evaluation scripts."""

from __future__ import annotations


def format_hcesvm_time_limit_message(time_limit_seconds: int | None) -> str:
    """Render the configured HCESVM per-classifier time limit without rewriting it."""
    if time_limit_seconds is None:
        return "HCESVM per-classifier time limit: none"
    return f"HCESVM per-classifier time limit: {int(time_limit_seconds)}s"
