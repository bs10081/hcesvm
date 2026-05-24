"""Tests for deadline-aware training decisions."""

from datetime import datetime, timezone

from hcesvm.utils.deadline_control import (
    estimate_remaining_seconds,
    evaluate_deadline_policy,
    parse_deadline,
)


def test_parse_deadline_converts_local_time_to_utc():
    local_deadline, utc_deadline = parse_deadline("2026-04-25T12:00:00", "Asia/Taipei")

    assert local_deadline.isoformat() == "2026-04-25T12:00:00+08:00"
    assert utc_deadline.isoformat() == "2026-04-25T04:00:00+00:00"


def test_estimate_remaining_seconds_uses_last_duration_when_it_is_larger():
    estimate = estimate_remaining_seconds([120.0, 240.0, 600.0], remaining_steps=2)

    assert estimate == 1200.0


def test_estimate_remaining_seconds_uses_median_when_it_is_larger():
    estimate = estimate_remaining_seconds([600.0, 240.0, 120.0], remaining_steps=2)

    assert estimate == 480.0


def test_deadline_policy_continues_before_deadline_even_if_estimate_is_large():
    decision = evaluate_deadline_policy(
        now_utc=datetime(2026, 4, 25, 3, 59, tzinfo=timezone.utc),
        deadline_utc=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
        overrun_buffer_minutes=30,
        completed_durations=[1800.0],
        remaining_steps=3,
    )

    assert decision.continue_training is True
    assert decision.decision == "continue"
    assert decision.deadline_reached is False


def test_deadline_policy_stops_after_deadline_when_estimate_exceeds_buffer():
    decision = evaluate_deadline_policy(
        now_utc=datetime(2026, 4, 25, 4, 5, tzinfo=timezone.utc),
        deadline_utc=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
        overrun_buffer_minutes=30,
        completed_durations=[1200.0],
        remaining_steps=2,
    )

    assert decision.continue_training is False
    assert decision.decision == "stop"
    assert decision.deadline_reached is True
    assert decision.estimated_finish_utc.isoformat() == "2026-04-25T04:45:00+00:00"


def test_deadline_policy_allows_completion_within_buffer_after_deadline():
    decision = evaluate_deadline_policy(
        now_utc=datetime(2026, 4, 25, 4, 5, tzinfo=timezone.utc),
        deadline_utc=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
        overrun_buffer_minutes=30,
        completed_durations=[300.0, 360.0],
        remaining_steps=2,
    )

    assert decision.continue_training is True
    assert decision.decision == "continue"
    assert decision.deadline_reached is True
    assert decision.estimated_finish_utc.isoformat() == "2026-04-25T04:17:00+00:00"
