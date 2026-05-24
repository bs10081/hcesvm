"""Helpers for deadline-aware training decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Sequence
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class DeadlineDecision:
    """Decision for whether training should continue after a checkpoint."""

    continue_training: bool
    decision: str
    deadline_reached: bool
    estimated_remaining_seconds: float | None
    estimated_finish_utc: datetime | None
    allowed_finish_utc: datetime


def parse_deadline(deadline_at: str, timezone_name: str) -> tuple[datetime, datetime]:
    """Parse a local wall-clock deadline and return local/UTC datetimes."""
    timezone_info = ZoneInfo(timezone_name)
    parsed = datetime.fromisoformat(deadline_at)

    if parsed.tzinfo is None:
        local_deadline = parsed.replace(tzinfo=timezone_info)
    else:
        local_deadline = parsed.astimezone(timezone_info)

    return local_deadline, local_deadline.astimezone(timezone.utc)


def estimate_remaining_seconds(
    completed_durations: Sequence[float],
    remaining_steps: int,
) -> float | None:
    """Estimate remaining solve time from completed step durations."""
    if remaining_steps <= 0:
        return 0.0
    if not completed_durations:
        return None

    completed = [float(duration) for duration in completed_durations]
    baseline_seconds = max(completed[-1], float(median(completed)))
    return float(remaining_steps) * baseline_seconds


def evaluate_deadline_policy(
    *,
    now_utc: datetime,
    deadline_utc: datetime,
    overrun_buffer_minutes: int,
    completed_durations: Sequence[float],
    remaining_steps: int,
) -> DeadlineDecision:
    """Apply the deadline policy used by the deadline-aware HCESVM runner."""
    allowed_finish_utc = deadline_utc + timedelta(minutes=overrun_buffer_minutes)
    deadline_reached = now_utc >= deadline_utc
    estimated_remaining_seconds = estimate_remaining_seconds(completed_durations, remaining_steps)

    if estimated_remaining_seconds is None:
        estimated_finish_utc = None
    else:
        estimated_finish_utc = now_utc + timedelta(seconds=estimated_remaining_seconds)

    if remaining_steps <= 0:
        return DeadlineDecision(
            continue_training=True,
            decision="complete",
            deadline_reached=deadline_reached,
            estimated_remaining_seconds=estimated_remaining_seconds,
            estimated_finish_utc=estimated_finish_utc,
            allowed_finish_utc=allowed_finish_utc,
        )

    if not deadline_reached:
        return DeadlineDecision(
            continue_training=True,
            decision="continue",
            deadline_reached=False,
            estimated_remaining_seconds=estimated_remaining_seconds,
            estimated_finish_utc=estimated_finish_utc,
            allowed_finish_utc=allowed_finish_utc,
        )

    if estimated_finish_utc is None:
        return DeadlineDecision(
            continue_training=False,
            decision="stop",
            deadline_reached=True,
            estimated_remaining_seconds=estimated_remaining_seconds,
            estimated_finish_utc=estimated_finish_utc,
            allowed_finish_utc=allowed_finish_utc,
        )

    continue_training = estimated_finish_utc <= allowed_finish_utc
    return DeadlineDecision(
        continue_training=continue_training,
        decision="continue" if continue_training else "stop",
        deadline_reached=True,
        estimated_remaining_seconds=estimated_remaining_seconds,
        estimated_finish_utc=estimated_finish_utc,
        allowed_finish_utc=allowed_finish_utc,
    )
