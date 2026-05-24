"""Helpers for the derived 1000-sample teaching-data HCESVM validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gurobipy import GRB


DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/gagolews/teaching-data/master/ordinal_regression"
)
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_TARGET_SIZE = 800
DEFAULT_TEST_TARGET_SIZE = 200


@dataclass(frozen=True, slots=True)
class DerivedDatasetRecipe:
    """Recipe for deriving a smaller teaching-data split from an existing split."""

    name: str
    source_dataset: str
    description: str
    train_target_size: int = DEFAULT_TRAIN_TARGET_SIZE
    test_target_size: int = DEFAULT_TEST_TARGET_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    class_map: Mapping[int, int] | None = None
    expected_train_counts: tuple[int, ...] | None = None
    expected_test_counts: tuple[int, ...] | None = None

    def with_overrides(self, **changes: Any) -> "DerivedDatasetRecipe":
        """Return a new recipe with one or more fields replaced."""
        return replace(self, **changes)


@dataclass(slots=True)
class DerivedDatasetSplit:
    """A derived train/test split with metadata needed for reports and runs."""

    name: str
    source_dataset: str
    source_url: str
    description: str
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
    class_map: dict[int, int] | None

    @property
    def n_classes(self) -> int:
        return len(self.train_original_counts)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def train_classes(self) -> list[np.ndarray]:
        return split_features_by_class(self.X_train, self.y_train, self.n_classes)

    @property
    def test_classes(self) -> list[np.ndarray]:
        return split_features_by_class(self.X_test, self.y_test, self.n_classes)


SKILL_METHOD3_4CLASS_1000_RECIPE = DerivedDatasetRecipe(
    name="skill_method3_4class_1000",
    source_dataset="skill",
    description="Keep original skill classes 4/5/6/7, relabel to 1/2/3/4, then downsample to train=800/test=200.",
    class_map={4: 1, 5: 2, 6: 3, 7: 4},
    expected_train_counts=(286, 283, 219, 12),
    expected_test_counts=(71, 71, 55, 3),
)

CALIFORNIAHOUSING_1000_RECIPE = DerivedDatasetRecipe(
    name="californiahousing_1000",
    source_dataset="californiahousing",
    description="Reuse the existing californiahousing 6-class split, then downsample to train=800/test=200.",
    expected_train_counts=(126, 265, 197, 99, 53, 60),
    expected_test_counts=(32, 66, 49, 25, 13, 15),
)

DERIVED_DATASET_RECIPES = {
    SKILL_METHOD3_4CLASS_1000_RECIPE.name: SKILL_METHOD3_4CLASS_1000_RECIPE,
    CALIFORNIAHOUSING_1000_RECIPE.name: CALIFORNIAHOUSING_1000_RECIPE,
}


def source_dataset_url(dataset_name: str) -> str:
    """Return the canonical teaching-data CSV URL for one source dataset."""
    return f"{DATASET_BASE_URL}/{dataset_name}.csv"


def default_feature_names(n_features: int) -> list[str]:
    """Create fallback feature names when the source loader does not expose them."""
    return [f"feature_{index}" for index in range(1, int(n_features) + 1)]


def class_counts(y: np.ndarray, *, n_classes: int) -> list[int]:
    """Count samples for consecutive class labels starting from 1."""
    return [int(np.sum(y == class_index)) for class_index in range(1, n_classes + 1)]


def split_features_by_class(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> list[np.ndarray]:
    """Split a feature matrix into one array per class label."""
    return [
        np.asarray(X[y == class_index], dtype=float)
        for class_index in range(1, n_classes + 1)
    ]


def filter_and_relabel_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_map: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter to selected classes and relabel them to consecutive integers."""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=int)
    if class_map is None:
        return X_array, y_array

    kept_labels = np.asarray(sorted(class_map), dtype=int)
    mask = np.isin(y_array, kept_labels)
    filtered_X = X_array[mask]
    filtered_y = np.asarray(
        [int(class_map[int(label)]) for label in y_array[mask]],
        dtype=int,
    )
    return filtered_X, filtered_y


def calculate_proportional_class_targets(
    class_sizes: list[int],
    target_size: int,
) -> list[int]:
    """Allocate a class-level target size with largest-remainder rounding."""
    counts = [int(size) for size in class_sizes]
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if sum(counts) < target_size:
        raise ValueError(
            f"target_size={target_size} exceeds available samples={sum(counts)}"
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
        raise ValueError(f"target_size={target_size} is too small for classes {counts}")
    return allocated


def sample_by_class_targets(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_targets: list[int],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rows within each class without replacement."""
    rng = np.random.default_rng(random_state)
    sampled_X: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []

    for class_index, target_size in enumerate(class_targets, start=1):
        class_rows = np.flatnonzero(y == class_index)
        if len(class_rows) < target_size:
            raise ValueError(
                f"class {class_index} only has {len(class_rows)} rows; cannot sample {target_size}"
            )
        chosen_rows = np.sort(rng.choice(class_rows, size=target_size, replace=False))
        sampled_X.append(np.asarray(X[chosen_rows], dtype=float))
        sampled_y.append(np.full(target_size, class_index, dtype=int))

    return np.vstack(sampled_X), np.concatenate(sampled_y)


def _validate_expected_counts(
    *,
    actual: list[int],
    expected: tuple[int, ...] | None,
    split_name: str,
    recipe_name: str,
) -> None:
    """Reject a recipe when the deterministic counts drift unexpectedly."""
    if expected is None:
        return
    if actual != list(expected):
        raise ValueError(
            f"{recipe_name} {split_name} counts drifted: expected {list(expected)}, got {actual}"
        )


def derive_dataset_split(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    source_url: str,
    recipe: DerivedDatasetRecipe,
    feature_names: list[str] | None = None,
) -> DerivedDatasetSplit:
    """Apply one derived-data recipe to an existing deterministic train/test split."""
    derived_feature_names = feature_names or default_feature_names(X_train.shape[1])
    filtered_X_train, filtered_y_train = filter_and_relabel_classes(
        X_train,
        y_train,
        class_map=recipe.class_map,
    )
    filtered_X_test, filtered_y_test = filter_and_relabel_classes(
        X_test,
        y_test,
        class_map=recipe.class_map,
    )

    n_classes = len(np.unique(filtered_y_train))
    train_original_counts = class_counts(filtered_y_train, n_classes=n_classes)
    test_original_counts = class_counts(filtered_y_test, n_classes=n_classes)
    train_targets = calculate_proportional_class_targets(
        train_original_counts,
        recipe.train_target_size,
    )
    test_targets = calculate_proportional_class_targets(
        test_original_counts,
        recipe.test_target_size,
    )
    _validate_expected_counts(
        actual=train_targets,
        expected=recipe.expected_train_counts,
        split_name="train",
        recipe_name=recipe.name,
    )
    _validate_expected_counts(
        actual=test_targets,
        expected=recipe.expected_test_counts,
        split_name="test",
        recipe_name=recipe.name,
    )

    sampled_X_train, sampled_y_train = sample_by_class_targets(
        filtered_X_train,
        filtered_y_train,
        class_targets=train_targets,
        random_state=recipe.random_state,
    )
    sampled_X_test, sampled_y_test = sample_by_class_targets(
        filtered_X_test,
        filtered_y_test,
        class_targets=test_targets,
        random_state=recipe.random_state + 1,
    )

    return DerivedDatasetSplit(
        name=recipe.name,
        source_dataset=recipe.source_dataset,
        source_url=source_url,
        description=recipe.description,
        feature_names=derived_feature_names,
        X_train=sampled_X_train,
        y_train=sampled_y_train,
        X_test=sampled_X_test,
        y_test=sampled_y_test,
        train_original_counts=train_original_counts,
        test_original_counts=test_original_counts,
        train_sampled_counts=train_targets,
        test_sampled_counts=test_targets,
        train_target_size=recipe.train_target_size,
        test_target_size=recipe.test_target_size,
        random_state=recipe.random_state,
        class_map=None if recipe.class_map is None else dict(recipe.class_map),
    )


def _write_split_csv(
    path: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*feature_names, "response"])
        for features, label in zip(X.tolist(), y.tolist()):
            writer.writerow([*features, int(label)])


def write_derived_split_artifacts(
    split: DerivedDatasetSplit,
    *,
    output_dir: Path,
    timestamp: str,
) -> dict[str, Path]:
    """Persist train/test CSVs plus a manifest for one derived split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / f"{split.name}_train_{timestamp}.csv"
    test_csv = output_dir / f"{split.name}_test_{timestamp}.csv"
    manifest_json = output_dir / f"{split.name}_manifest_{timestamp}.json"

    _write_split_csv(
        train_csv,
        X=split.X_train,
        y=split.y_train,
        feature_names=split.feature_names,
    )
    _write_split_csv(
        test_csv,
        X=split.X_test,
        y=split.y_test,
        feature_names=split.feature_names,
    )

    manifest = {
        "dataset": split.name,
        "source_dataset": split.source_dataset,
        "source_url": split.source_url,
        "description": split.description,
        "class_map": split.class_map,
        "random_state": split.random_state,
        "train_original_counts": split.train_original_counts,
        "test_original_counts": split.test_original_counts,
        "train_sampled_counts": split.train_sampled_counts,
        "test_sampled_counts": split.test_sampled_counts,
        "train_target_size": split.train_target_size,
        "test_target_size": split.test_target_size,
        "feature_names": split.feature_names,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "manifest_json": manifest_json,
    }


def format_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python values for logs and workbooks."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def format_vector(values: Any) -> str:
    """Serialize a vector compactly for logs, reports, and Excel output."""
    if values is None:
        return ""
    array = np.asarray(values, dtype=float).tolist()
    return json.dumps([round(float(value), 10) for value in array], ensure_ascii=False)


def solver_status_label(status_code: int | None, *, has_solution: bool) -> str:
    """Normalize raw Gurobi status codes to the labels required by the report."""
    if status_code == GRB.OPTIMAL:
        return "optimal"
    if status_code == GRB.TIME_LIMIT and has_solution:
        return "time_limit_with_solution"
    if status_code == GRB.MEM_LIMIT and has_solution:
        return "mem_limit_with_solution"
    return "failed_or_no_solution"


def build_classifier_diagnostics_row(
    *,
    dataset_name: str,
    progress: dict[str, Any],
    classifier: Any,
) -> dict[str, Any]:
    """Create one classifier diagnostics row with the required status fields."""
    solution = {} if classifier.solution is None else dict(classifier.solution)
    status_code_raw = solution.get("solver_status")
    status_code = None if status_code_raw is None else int(status_code_raw)
    status_label = solver_status_label(
        status_code,
        has_solution=classifier.solution is not None,
    )
    return {
        "dataset": dataset_name,
        "classifier": f"H{int(progress['hk'])}",
        "description": progress["description"],
        "positive_sample_count": int(progress["positive_sample_count"]),
        "negative_sample_count": int(progress["negative_sample_count"]),
        "weights": format_vector(classifier.weights),
        "b": format_scalar(classifier.intercept),
        "objective_value": format_scalar(solution.get("objective_value")),
        "mip_gap": format_scalar(solution.get("mip_gap")),
        "solver_status_code": status_code,
        "solver_status_label": status_label,
        "mem_used_gb": format_scalar(solution.get("mem_used_gb")),
        "max_mem_used_gb": format_scalar(solution.get("max_mem_used_gb")),
        "elapsed_seconds": format_scalar(progress.get("elapsed_seconds")),
        "cumulative_elapsed_seconds": format_scalar(
            progress.get("cumulative_elapsed_seconds")
        ),
    }


def build_classifier_progress_row(
    *,
    diagnostics_row: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    """Create a progress worksheet row from one diagnostics record."""
    return {
        **diagnostics_row,
        "started_at_utc": progress.get("started_at_utc"),
        "finished_at_utc": progress.get("finished_at_utc"),
    }
