"""Runtime helpers for teaching-data evaluation scripts."""

from __future__ import annotations

from typing import Any


HCESVM_SPECIAL_PER_CLASSIFIER_LIMITS = {
    "skill": 3600,
    "californiahousing": 3600,
}


def resolve_three_model_hcesvm_time_limit(
    dataset_name: str,
    requested_total_time_limit: int,
    n_classes: int,
) -> tuple[int, str]:
    """Resolve the actual HCESVM per-classifier time limit for the three-model runner."""
    override = HCESVM_SPECIAL_PER_CLASSIFIER_LIMITS.get(dataset_name)
    if override is not None:
        return (
            override,
            f"Dataset-specific HCESVM per-classifier time limit override: {override}s "
            f"(no total-budget split) for {dataset_name}",
        )

    subproblem_count = max(int(n_classes) - 1, 1)
    per_classifier_time_limit = max(1, int(requested_total_time_limit) // subproblem_count)
    return (
        per_classifier_time_limit,
        f"Total HCESVM time budget: {int(requested_total_time_limit)}s; "
        f"per binary classifier time limit: {per_classifier_time_limit}s "
        f"across {subproblem_count} classifiers",
    )


def resolve_deadline_runner_hcesvm_time_limit(
    dataset_name: str,
    requested_time_limit: int | None,
) -> tuple[int | None, str]:
    """Resolve the actual HCESVM per-classifier time limit for the deadline runner."""
    if requested_time_limit is None:
        return None, "HCESVM per-classifier time limit: none"

    override = HCESVM_SPECIAL_PER_CLASSIFIER_LIMITS.get(dataset_name)
    if override is not None:
        return (
            override,
            f"Dataset-specific HCESVM per-classifier time limit override: {override}s for {dataset_name}",
        )

    return requested_time_limit, f"HCESVM per-classifier time limit: {int(requested_time_limit)}s"


def encode_special_limit_metadata() -> str:
    """Return a compact metadata string for special HCESVM teaching-data limits."""
    parts = [
        f"{dataset_name}:{time_limit_seconds}"
        for dataset_name, time_limit_seconds in sorted(HCESVM_SPECIAL_PER_CLASSIFIER_LIMITS.items())
    ]
    return ", ".join(parts)
