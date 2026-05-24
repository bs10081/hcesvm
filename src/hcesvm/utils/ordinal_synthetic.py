"""Synthetic ordinal datasets for demos and tests."""

from __future__ import annotations

from typing import Sequence


def make_staircase_ordinal_data(
    *,
    num_classes: int = 5,
    samples_per_class: int = 5,
    labels: Sequence[int] | None = None,
    gap: float = 6.0,
    two_dimensional: bool = True,
) -> tuple[list[list[float]], list[int]]:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1.")

    if labels is None:
        label_values = [class_index + 1 for class_index in range(num_classes)]
    else:
        label_values = list(labels)
        if len(label_values) != num_classes:
            raise ValueError("labels must have exactly num_classes elements.")

    offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    if samples_per_class > len(offsets):
        raise ValueError("samples_per_class is capped at 5 in this deterministic generator.")

    X: list[list[float]] = []
    y: list[int] = []
    midpoint = (num_classes - 1) / 2

    for class_index, label in enumerate(label_values):
        center = (class_index - midpoint) * gap
        for sample_index in range(samples_per_class):
            offset = offsets[sample_index]
            if two_dimensional:
                row = [center + offset, 0.5 * center - 0.25 * offset]
            else:
                row = [center + offset]
            X.append(row)
            y.append(label)

    return X, y
