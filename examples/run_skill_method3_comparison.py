#!/usr/bin/env python3
"""Run Method 3 on the derived 4-class skill split and compare against SVOR/NPSVOR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.run_teaching_data_three_models import (
    DEFAULT_HCESVM_PARAMS,
    DEFAULT_NPSVOR_PARAMS,
    DEFAULT_SVOR_PARAMS,
    DatasetBundle,
    TeeStream,
    banner,
    checkpoint_excel,
    collect_metric_row,
    git_output,
    human_timestamp,
    load_dataset,
    run_hcesvm,
    run_npsvor,
    run_svor,
    short_timestamp,
    utc_now,
    write_excel,
)
from hcesvm.utils.skill_method3 import (
    DEFAULT_METHOD3_RANDOM_STATE,
    DEFAULT_METHOD3_TEST_TARGET,
    DEFAULT_METHOD3_TRAIN_TARGET,
    METHOD3_DATASET_NAME,
    build_method3_metadata_rows,
    derive_skill_method3_split,
    write_method3_artifacts,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the derived Method 3 skill 4-class comparison against SVOR and NPSVOR."
    )
    parser.add_argument(
        "--train-target-size",
        type=int,
        default=DEFAULT_METHOD3_TRAIN_TARGET,
        help=f"Derived training size after proportional downsampling. Default: {DEFAULT_METHOD3_TRAIN_TARGET}.",
    )
    parser.add_argument(
        "--test-target-size",
        type=int,
        default=DEFAULT_METHOD3_TEST_TARGET,
        help=f"Derived testing size after proportional downsampling. Default: {DEFAULT_METHOD3_TEST_TARGET}.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_METHOD3_RANDOM_STATE,
        help=f"Random seed for reproducible Method 3 sampling. Default: {DEFAULT_METHOD3_RANDOM_STATE}.",
    )
    parser.add_argument(
        "--hcesvm-time-limit",
        type=int,
        default=DEFAULT_HCESVM_PARAMS["time_limit"],
        help="HCESVM per-classifier time limit in seconds.",
    )
    parser.add_argument(
        "--svor-time-limit",
        type=int,
        default=DEFAULT_SVOR_PARAMS["solver_params"]["TimeLimit"],
        help="SVOR total time budget in seconds.",
    )
    parser.add_argument(
        "--npsvor-time-limit",
        type=int,
        default=DEFAULT_NPSVOR_PARAMS["solver_params"]["TimeLimit"],
        help="NPSVOR total time budget in seconds.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_HCESVM_PARAMS["threads"],
        help=f"Gurobi thread count for HCESVM. Default: {DEFAULT_HCESVM_PARAMS['threads']}.",
    )
    parser.add_argument(
        "--soft-mem-limit-gb",
        type=float,
        default=DEFAULT_HCESVM_PARAMS["soft_mem_limit_gb"],
        help=(
            "Gurobi SoftMemLimit for HCESVM in GB. "
            f"Default: {DEFAULT_HCESVM_PARAMS['soft_mem_limit_gb']}."
        ),
    )
    parser.add_argument(
        "--dataset-label",
        default=METHOD3_DATASET_NAME,
        help=f"Derived dataset label. Default: {METHOD3_DATASET_NAME}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    started_at = utc_now()
    timestamp = short_timestamp(started_at)
    date_prefix = started_at.strftime("%Y%m%d")
    archive_dir = ROOT / "results" / "archive" / f"{date_prefix}_method3_skill_4class"
    archive_dir.mkdir(parents=True, exist_ok=True)

    log_path = archive_dir / f"method3_skill_4class_cementscale_{timestamp}.log"
    xlsx_path = ROOT / "docs" / "reports" / f"SKILL_METHOD3_4CLASS_CEMENTSCALE_{timestamp}.xlsx"

    hcesvm_params = dict(DEFAULT_HCESVM_PARAMS)
    hcesvm_params["time_limit"] = args.hcesvm_time_limit
    hcesvm_params["threads"] = args.threads
    hcesvm_params["soft_mem_limit_gb"] = args.soft_mem_limit_gb

    svor_params = dict(DEFAULT_SVOR_PARAMS)
    svor_params["solver_params"] = dict(DEFAULT_SVOR_PARAMS["solver_params"])
    svor_params["solver_params"]["TimeLimit"] = args.svor_time_limit

    npsvor_params = dict(DEFAULT_NPSVOR_PARAMS)
    npsvor_params["solver_params"] = dict(DEFAULT_NPSVOR_PARAMS["solver_params"])
    npsvor_params["solver_params"]["TimeLimit"] = args.npsvor_time_limit

    source_bundle = load_dataset("skill")
    split = derive_skill_method3_split(
        X_train=source_bundle.X_train,
        y_train=source_bundle.y_train,
        X_test=source_bundle.X_test,
        y_test=source_bundle.y_test,
        source_url=source_bundle.source_url,
        dataset_name=args.dataset_label,
        train_target_size=args.train_target_size,
        test_target_size=args.test_target_size,
        random_state=args.random_state,
    )
    derived_paths = write_method3_artifacts(split, output_dir=archive_dir, timestamp=timestamp)
    bundle = DatasetBundle(
        name=split.name,
        source_url=split.source_url,
        n_classes=split.n_classes,
        n_features=split.n_features,
        train_classes=split.train_classes,
        test_classes=split.test_classes,
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
    )

    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        tee = TeeStream(sys.stdout, log_handle)
        banner(tee, "Method 3 skill 4-class comparison")
        print(f"Branch: {git_output('git', 'rev-parse', '--abbrev-ref', 'HEAD')}", file=tee)
        print(f"Commit: {git_output('git', 'rev-parse', 'HEAD')}", file=tee)
        print("Source dataset: skill", file=tee)
        print("Method 3 relabel map: 4->1, 5->2, 6->3, 7->4", file=tee)
        print(
            "Split rule: reuse skill stratified 80/20 split, then proportionally downsample train/test independently",
            file=tee,
        )
        print(f"Sampling seed: {args.random_state}", file=tee)
        print(
            f"Train counts before/after: {split.train_original_counts} -> {split.train_sampled_counts}",
            file=tee,
        )
        print(
            f"Test counts before/after: {split.test_original_counts} -> {split.test_sampled_counts}",
            file=tee,
        )
        print(f"Derived train CSV: {derived_paths['train_csv']}", file=tee)
        print(f"Derived test CSV: {derived_paths['test_csv']}", file=tee)
        print(f"Manifest JSON: {derived_paths['manifest_json']}", file=tee)
        print(f"HCESVM params: {json.dumps(hcesvm_params, ensure_ascii=False)}", file=tee)
        print(f"SVOR params: {json.dumps(svor_params, ensure_ascii=False)}", file=tee)
        print(f"NPSVOR params: {json.dumps(npsvor_params, ensure_ascii=False)}", file=tee)

        metadata_rows = build_method3_metadata_rows(
            split,
            log_path=log_path,
            excel_path=xlsx_path,
            derived_paths=derived_paths,
            hcesvm_params=hcesvm_params,
            svor_params=svor_params,
            npsvor_params=npsvor_params,
            started_at_utc=human_timestamp(started_at),
            git_branch=git_output("git", "rev-parse", "--abbrev-ref", "HEAD"),
            git_commit=git_output("git", "rev-parse", "HEAD"),
            worktree=str(ROOT),
        )
        checkpoint_excel(xlsx_path, metric_rows, parameter_rows, metadata_rows, tee, "initialization")

        banner(tee, f"Dataset {bundle.name}")
        print(f"Source: {bundle.source_url}", file=tee)
        print(f"Classes: {bundle.n_classes}", file=tee)
        print(f"Features: {bundle.n_features}", file=tee)
        print(f"Train counts: {bundle.train_counts}", file=tee)
        print(f"Test counts: {bundle.test_counts}", file=tee)

        results = (
            run_hcesvm(bundle, tee, hcesvm_params),
            run_svor(bundle, tee, svor_params),
            run_npsvor(bundle, tee, npsvor_params),
        )

        for result in results:
            metric_rows.append(collect_metric_row(bundle, result))
            parameter_rows.extend(result.parameter_rows)
            checkpoint_excel(
                xlsx_path,
                metric_rows,
                parameter_rows,
                metadata_rows,
                tee,
                f"{bundle.name} / {result.model}",
            )

        banner(tee, "Run complete")
        print(f"Log saved to: {log_path}", file=tee)
        print(f"Excel report target: {xlsx_path}", file=tee)

    write_excel(xlsx_path, metric_rows, parameter_rows, metadata_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
