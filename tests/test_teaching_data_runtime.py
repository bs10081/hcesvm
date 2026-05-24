"""Tests for teaching-data runtime configuration helpers."""

from hcesvm.utils.teaching_data_runtime import format_hcesvm_time_limit_message


def test_format_hcesvm_time_limit_message_renders_none_without_override():
    assert format_hcesvm_time_limit_message(None) == "HCESVM per-classifier time limit: none"


def test_format_hcesvm_time_limit_message_preserves_explicit_per_classifier_value():
    assert format_hcesvm_time_limit_message(1800) == "HCESVM per-classifier time limit: 1800s"
