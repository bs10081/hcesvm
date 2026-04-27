"""Tests for teaching-data runtime configuration helpers."""

from hcesvm.utils.teaching_data_runtime import (
    encode_special_limit_metadata,
    resolve_deadline_runner_hcesvm_time_limit,
    resolve_three_model_hcesvm_time_limit,
)


def test_three_model_runner_uses_dataset_override_for_skill():
    limit, message = resolve_three_model_hcesvm_time_limit("skill", requested_total_time_limit=1800, n_classes=7)

    assert limit == 3600
    assert "override" in message
    assert "skill" in message


def test_three_model_runner_splits_total_budget_for_normal_dataset():
    limit, message = resolve_three_model_hcesvm_time_limit(
        "stock_ord",
        requested_total_time_limit=1800,
        n_classes=5,
    )

    assert limit == 450
    assert "per binary classifier time limit: 450s across 4 classifiers" in message


def test_deadline_runner_keeps_none_when_time_limit_is_unset_even_for_override_dataset():
    limit, message = resolve_deadline_runner_hcesvm_time_limit("californiahousing", requested_time_limit=None)

    assert limit is None
    assert message == "HCESVM per-classifier time limit: none"


def test_deadline_runner_uses_dataset_override_when_time_limit_is_requested():
    limit, message = resolve_deadline_runner_hcesvm_time_limit("californiahousing", requested_time_limit=1800)

    assert limit == 3600
    assert "californiahousing" in message


def test_deadline_runner_keeps_none_for_non_override_dataset():
    limit, message = resolve_deadline_runner_hcesvm_time_limit("cement_strength", requested_time_limit=None)

    assert limit is None
    assert message == "HCESVM per-classifier time limit: none"


def test_special_limit_metadata_is_stable():
    assert encode_special_limit_metadata() == "californiahousing:3600, skill:3600"
