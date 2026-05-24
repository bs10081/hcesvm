"""Helpers for the derived 4-class Method 3 skill experiment."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


METHOD3_SOURCE_DATASET = "skill"
METHOD3_DATASET_NAME = "skill_method3_4to7_cementscale"
METHOD3_KEEP_CLASS_MAP = {
    4: 1,
    5: 2,
    6: 3,
    7: 4,
}
DEFAULT_METHOD3_TRAIN_TARGET = 798
DEFAULT_METHOD3_TEST_TARGET = 200
DEFAULT_METHOD3_RANDOM_STATE = 42


@dataclass(slots=True)
class DerivedSkillMethod3Split:
    """A derived train/test split for the Method 3 skill comparison."""

    name: str
    source_dataset: str
    source_url: str
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_original_counts: list[int]
    test_original_counts: list[int]
    train_sampled_counts: list[int]
    test_sampled_counts: list[int]
    train_target_size: int
    test_target_size: int
    random_state: int

    @property
    def n_classes(self) -> int:
        return len(METHOD3_KEEP_CLASS_MAP)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def train_classes(self) -> list[np.ndarray]:
        return split_features_by_class(self.X_train, self.y_train, self.n_classes)

    @property
    def test_classes(self) -> list[np.ndarray]:
        return split_features_by_class(self.X_test, self.y_test, self.n_classes)


def default_feature_names(n_features: int) -> list[str]:
    return [f"feature_{index}" for index in range(1, int(n_features) + 1)]


def class_counts(y: np.ndarray, *, n_classes: int) -> list[int]:
    return [int(np.sum(y == class_index)) for class_index in range(1, n_classes + 1)]


def filter_and_relabel_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    keep_class_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the requested ordinal classes and relabel them consecutively."""
    mapping = keep_class_map or METHOD3_KEEP_CLASS_MAP
    kept_labels = np.asarray(sorted(mapping), dtype=int)
    mask = np.isin(y, kept_labels)
    filtered_X = np.asarray(X[mask], dtype=float)
    filtered_y = np.asarray([mapping[int(label)] for label in y[mask]], dtype=int)
    return filtered_X, filtered_y


def calculate_proportional_class_targets(class_sizes: list[int], target_size: int) -> list[int]:
    """Allocate a target sample size proportionally with largest-remainder rounding."""
    counts = [int(size) for size in class_sizes]
    if target_size <= 0:
        raise ValueError("target_size must be positive.")
    if sum(counts) < target_size:
        raise ValueError(
            f"target_size={target_size} exceeds available samples={sum(counts)}."
        )

    total = sum(counts)
    raw_targets = [(count * target_size) / total for count in counts]
    allocated = [int(np.floor(value)) for value in raw_targets]

    remainder = int(target_size - sum(allocated))
    ranked_indices = sorted(
        range(len(counts)),
        key=lambda index: (raw_targets[index] - allocated[index], counts[index]),
        reverse=True,
    )
    for index in ranked_indices[:remainder]:
        allocated[index] += 1

    if any(value <= 0 for value in allocated):
        raise ValueError(
            "target_size is too small to keep all Method 3 classes. "
            f"Allocated counts: {allocated!r}"
        )

    return allocated


def sample_by_class_targets(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_targets: list[int],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rows within each class without replacement using fixed targets."""
    rng = np.random.default_rng(random_state)
    sampled_X: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []

    for class_index, target_size in enumerate(class_targets, start=1):
        class_rows = np.flatnonzero(y == class_index)
        if len(class_rows) < target_size:
            raise ValueError(
                f"Class {class_index} only has {len(class_rows)} rows; cannot sample {target_size}."
            )

        chosen_rows = np.sort(rng.choice(class_rows, size=target_size, replace=False))
        sampled_X.append(np.asarray(X[chosen_rows], dtype=float))
        sampled_y.append(np.full(target_size, class_index, dtype=int))

    return np.vstack(sampled_X), np.concatenate(sampled_y)


def split_features_by_class(X: np.ndarray, y: np.ndarray, n_classes: int) -> list[np.ndarray]:
    return [np.asarray(X[y == class_index], dtype=float) for class_index in range(1, n_classes + 1)]


def derive_skill_method3_split(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    source_url: str,
    dataset_name: str = METHOD3_DATASET_NAME,
    source_dataset: str = METHOD3_SOURCE_DATASET,
    feature_names: list[str] | None = None,
    train_target_size: int = DEFAULT_METHOD3_TRAIN_TARGET,
    test_target_size: int = DEFAULT_METHOD3_TEST_TARGET,
    random_state: int = DEFAULT_METHOD3_RANDOM_STATE,
) -> DerivedSkillMethod3Split:
    """Build the fixed Method 3 derived split from an existing skill split."""
    method3_feature_names = feature_names or default_feature_names(X_train.shape[1])
    filtered_X_train, filtered_y_train = filter_and_relabel_classes(np.asarray(X_train), np.asarray(y_train))
    filtered_X_test, filtered_y_test = filter_and_relabel_classes(np.asarray(X_test), np.asarray(y_test))

    train_original_counts = class_counts(filtered_y_train, n_classes=len(METHOD3_KEEP_CLASS_MAP))
    test_original_counts = class_counts(filtered_y_test, n_classes=len(METHOD3_KEEP_CLASS_MAP))
    train_targets = calculate_proportional_class_targets(train_original_counts, train_target_size)
    test_targets = calculate_proportional_class_targets(test_original_counts, test_target_size)

    sampled_X_train, sampled_y_train = sample_by_class_targets(
        filtered_X_train,
        filtered_y_train,
        class_targets=train_targets,
        random_state=random_state,
    )
    sampled_X_test, sampled_y_test = sample_by_class_targets(
        filtered_X_test,
        filtered_y_test,
        class_targets=test_targets,
        random_state=random_state + 1,
    )

    return DerivedSkillMethod3Split(
        name=dataset_name,
        source_dataset=source_dataset,
        source_url=source_url,
        feature_names=method3_feature_names,
        X_train=sampled_X_train,
        y_train=sampled_y_train,
        X_test=sampled_X_test,
        y_test=sampled_y_test,
        train_original_counts=train_original_counts,
        test_original_counts=test_original_counts,
        train_sampled_counts=train_targets,
        test_sampled_counts=test_targets,
        train_target_size=train_target_size,
        test_target_size=test_target_size,
        random_state=random_state,
    )


def _write_split_csv(path: Path, *, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*feature_names, "response"])
        for features, label in zip(X.tolist(), y.tolist()):
            writer.writerow([*features, int(label)])


def write_method3_artifacts(
    split: DerivedSkillMethod3Split,
    *,
    output_dir: Path,
    timestamp: str,
) -> dict[str, Path]:
    """Persist derived train/test CSVs plus a compact manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / f"{split.name}_train_{timestamp}.csv"
    test_csv = output_dir / f"{split.name}_test_{timestamp}.csv"
    manifest_json = output_dir / f"{split.name}_manifest_{timestamp}.json"

    _write_split_csv(train_csv, X=split.X_train, y=split.y_train, feature_names=split.feature_names)
    _write_split_csv(test_csv, X=split.X_test, y=split.y_test, feature_names=split.feature_names)

    manifest = {
        "dataset": split.name,
        "source_dataset": split.source_dataset,
        "source_url": split.source_url,
        "keep_class_map": METHOD3_KEEP_CLASS_MAP,
        "random_state": split.random_state,
        "train_target_size": split.train_target_size,
        "test_target_size": split.test_target_size,
        "train_original_counts": split.train_original_counts,
        "test_original_counts": split.test_original_counts,
        "train_sampled_counts": split.train_sampled_counts,
        "test_sampled_counts": split.test_sampled_counts,
        "feature_names": split.feature_names,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "manifest_json": manifest_json,
    }


def build_method3_metadata_rows(
    split: DerivedSkillMethod3Split,
    *,
    log_path: Path,
    excel_path: Path,
    derived_paths: dict[str, Path],
    hcesvm_params: dict[str, Any],
    svor_params: dict[str, Any],
    npsvor_params: dict[str, Any],
    started_at_utc: str,
    git_branch: str,
    git_commit: str,
    worktree: str,
) -> list[tuple[str, Any]]:
    """Create workbook metadata rows for the Method 3 run."""
    rows = [
        ("generated_at_utc", started_at_utc),
        ("branch", git_branch),
        ("commit", git_commit),
        ("worktree", worktree),
        ("log_path", str(log_path)),
        ("excel_path", str(excel_path)),
        ("dataset", split.name),
        ("source_dataset", split.source_dataset),
        ("source_url", split.source_url),
        ("method3_keep_class_map", json.dumps(METHOD3_KEEP_CLASS_MAP, ensure_ascii=False)),
        ("random_state", split.random_state),
        ("split_rule", "Reuse existing skill stratified split; proportionally downsample train/test independently"),
        ("train_original_counts", json.dumps(split.train_original_counts, ensure_ascii=False)),
        ("test_original_counts", json.dumps(split.test_original_counts, ensure_ascii=False)),
        ("train_sampled_counts", json.dumps(split.train_sampled_counts, ensure_ascii=False)),
        ("test_sampled_counts", json.dumps(split.test_sampled_counts, ensure_ascii=False)),
        ("train_target_size", split.train_target_size),
        ("test_target_size", split.test_target_size),
        ("train_csv", str(derived_paths["train_csv"])),
        ("test_csv", str(derived_paths["test_csv"])),
        ("manifest_json", str(derived_paths["manifest_json"])),
        ("hcesvm_params", json.dumps(hcesvm_params, ensure_ascii=False)),
        ("svor_params", json.dumps(svor_params, ensure_ascii=False)),
        ("npsvor_params", json.dumps(npsvor_params, ensure_ascii=False)),
    ]
    return rows
