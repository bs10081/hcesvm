"""Tests for teaching-data post-run analysis helpers."""

from hcesvm.utils.postrun_analysis import (
    evaluate_significant_improvement,
    format_duration_label,
    historical_three_model_per_classifier_limit,
    recommend_time_limit_seconds,
)


def test_historical_three_model_per_classifier_limit_recovers_old_budget_split():
    assert historical_three_model_per_classifier_limit(7) == 300
    assert historical_three_model_per_classifier_limit(6) == 360


def test_evaluate_significant_improvement_flags_total_and_zero_accuracy_gains():
    baseline_metrics = {
        "test_total_accuracy": 0.24,
        "test_class_1_accuracy": 0.0,
        "test_class_2_accuracy": 0.0,
        "test_class_3_accuracy": 0.53,
        "test_class_4_accuracy": 0.64,
        "test_class_5_accuracy": 0.0,
        "test_class_6_accuracy": 0.0,
        "test_class_7_accuracy": 0.0,
    }
    candidate_metrics = {
        "test_total_accuracy": 0.29,
        "test_class_1_accuracy": 0.1,
        "test_class_2_accuracy": 0.2,
        "test_class_3_accuracy": 0.55,
        "test_class_4_accuracy": 0.66,
        "test_class_5_accuracy": 0.05,
        "test_class_6_accuracy": 0.0,
        "test_class_7_accuracy": 0.0,
    }

    significant, reason, details = evaluate_significant_improvement(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        n_classes=7,
    )

    assert significant is True
    assert "test total accuracy" in reason
    assert details["zero_test_classes_reduced"] == 3


def test_recommend_time_limit_seconds_keeps_runtime_headroom_when_improvement_is_significant():
    recommendation, reason = recommend_time_limit_seconds(
        baseline_limit_seconds=300,
        current_limit_seconds=3600,
        progress_rows=[{"elapsed_seconds": 1610.0}, {"elapsed_seconds": 1330.0}],
        final_status="completed",
        significant_improvement=True,
        delta_test_total_accuracy=0.04,
        delta_test_macro_accuracy=0.06,
        zero_test_classes_reduced=2,
    )

    assert recommendation == 2100
    assert "accuracy improved materially" in reason


def test_recommend_time_limit_seconds_falls_back_to_balanced_shorter_limit_when_no_gain():
    recommendation, reason = recommend_time_limit_seconds(
        baseline_limit_seconds=360,
        current_limit_seconds=3600,
        progress_rows=[{"elapsed_seconds": 3500.0}],
        final_status="completed",
        significant_improvement=False,
        delta_test_total_accuracy=0.0,
        delta_test_macro_accuracy=-0.01,
        zero_test_classes_reduced=0,
    )

    assert recommendation == 900
    assert "no meaningful accuracy gain" in reason


def test_format_duration_label_rounds_to_minutes():
    assert format_duration_label(300) == "5m"
    assert format_duration_label(2100) == "35m"
    assert format_duration_label(3660) == "1h 1m"
