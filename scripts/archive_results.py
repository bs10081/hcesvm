#!/usr/bin/env python3
"""
HCESVM Results Archival Script

Archives test result logs from results/ directory into dated subdirectories.

Usage:
    python scripts/archive_results.py [--dry-run] [--yes]

Options:
    --dry-run    Show what would be archived without actually moving files (default)
    --execute    Actually execute the archival
    --yes        Skip confirmation prompts (use with caution!)

Examples:
    # Preview what would be archived
    python scripts/archive_results.py

    # Execute archival with confirmation
    python scripts/archive_results.py --execute

    # Execute archival without confirmation
    python scripts/archive_results.py --execute --yes

Archival Structure:
    results/archive/
    ├── 20260130_inverted/
    ├── 20260131_inverted/
    ├── 20260201_test2/
    ├── 20260203_test2_fixed/
    ├── 20260210_comparison/
    ├── 20260211_test3/
    ├── 20260213_class1_first/
    └── 20260224_class1_first/
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# Pattern to extract date and strategy from log filename
# Format: {strategy}_{dataset}_{YYYYMMDD}_{HHMMSS}.log
LOG_PATTERN = re.compile(r'^(.*?)_(\d{8})_\d{6}\.log$')


def parse_log_filename(filename):
    """
    Parse log filename to extract strategy and date.

    Args:
        filename: Log filename (e.g., "inverted_Abalone_20260130_173416.log")

    Returns:
        tuple: (strategy, date_str) or (None, None) if no match
    """
    match = LOG_PATTERN.match(filename)
    if match:
        # Extract everything before the date as strategy+dataset
        prefix = match.group(1)
        date_str = match.group(2)  # YYYYMMDD

        # Determine strategy from prefix
        if prefix.startswith('inverted_'):
            strategy = 'inverted'
        elif prefix.startswith('test2_fixed_'):
            strategy = 'test2_fixed'
        elif prefix.startswith('test2_'):
            # Check if it might be a comparison (both test2 and test3 on same date)
            strategy = 'test2'
        elif prefix.startswith('test3_'):
            strategy = 'test3'
        elif prefix.startswith('class1_first_'):
            strategy = 'class1_first'
        else:
            strategy = 'unknown'

        return strategy, date_str

    return None, None


def group_logs_by_date_strategy(log_files):
    """
    Group log files by date and strategy.

    Returns:
        dict: {(date, strategy): [file_paths]}
    """
    grouped = defaultdict(list)

    for log_path in log_files:
        filename = os.path.basename(log_path)
        strategy, date_str = parse_log_filename(filename)

        if strategy and date_str:
            grouped[(date_str, strategy)].append(log_path)

    return grouped


def determine_archive_name(date_str, strategy, files):
    """
    Determine archive directory name.

    Args:
        date_str: Date in YYYYMMDD format
        strategy: Strategy name
        files: List of file paths in this group

    Returns:
        str: Archive directory name (e.g., "20260130_inverted")
    """
    # Check if this might be a comparison (both test2 and test3 on same date)
    has_test2 = any('test2_' in os.path.basename(f) for f in files)
    has_test3 = any('test3_' in os.path.basename(f) for f in files)

    if has_test2 and has_test3:
        return f"{date_str}_comparison"
    else:
        return f"{date_str}_{strategy}"


def scan_results_logs(results_dir):
    """
    Scan results/ directory for logs that should be archived.

    Returns:
        dict: {archive_name: [file_paths]}
    """
    results_path = Path(results_dir)

    # Get all .log files in results/ (not in archive/)
    log_files = []
    for log_file in results_path.glob('*.log'):
        if 'archive' not in str(log_file):
            log_files.append(str(log_file))

    # Group by date and strategy
    grouped = group_logs_by_date_strategy(log_files)

    # Determine archive names
    archives = {}
    for (date_str, strategy), files in grouped.items():
        archive_name = determine_archive_name(date_str, strategy, files)
        if archive_name not in archives:
            archives[archive_name] = []
        archives[archive_name].extend(files)

    return archives


def display_archive_report(archives):
    """Display a formatted report of archival plan."""
    if not archives:
        print("✅ No logs found to archive. Results directory is up to date!")
        return

    total_files = sum(len(files) for files in archives.values())

    print("\n" + "="*70)
    print(" 📦 ARCHIVAL PLAN")
    print("="*70)
    print(f"\nFound {total_files} log file(s) to archive into {len(archives)} directory(ies)\n")

    for archive_name in sorted(archives.keys()):
        files = archives[archive_name]
        print(f"  📁 {archive_name}/ ({len(files)} file(s))")
        for file_path in sorted(files):
            filename = os.path.basename(file_path)
            size = os.path.getsize(file_path)
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            print(f"      📄 {filename:50s} {size_str:>10s}")
        print()

    print("="*70)


def archive_logs(archives, results_dir, dry_run=True, auto_yes=False):
    """
    Archive logs into dated subdirectories.

    Args:
        archives: Dict of {archive_name: [file_paths]}
        results_dir: Path to results directory
        dry_run: If True, only show what would be done
        auto_yes: If True, skip confirmation prompt
    """
    if not archives:
        print("✅ No logs to archive.")
        return

    if dry_run:
        print("\n[DRY RUN MODE] No files were actually moved.")
        print("Run with --execute flag to perform actual archival.\n")
        return

    # Confirmation
    if not auto_yes:
        total_files = sum(len(files) for files in archives.values())
        print(f"\n⚠️  This will move {total_files} log file(s) into archive subdirectories.")
        response = input("Proceed with archival? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Archival cancelled by user.")
            return

    # Execute archival
    print("\n📦 Executing archival...\n")
    results_path = Path(results_dir)
    archive_base = results_path / 'archive'
    archive_base.mkdir(exist_ok=True)

    success_count = 0
    error_count = 0

    for archive_name, files in sorted(archives.items()):
        archive_dir = archive_base / archive_name
        archive_dir.mkdir(exist_ok=True)
        print(f"  📁 {archive_name}/")

        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                dest_path = archive_dir / filename
                shutil.move(file_path, dest_path)
                print(f"      ✅ {filename}")
                success_count += 1
            except Exception as e:
                print(f"      ❌ Error moving {os.path.basename(file_path)}: {e}")
                error_count += 1
        print()

    print("="*70)
    print(f"✅ Archival complete!")
    print(f"   Archived: {success_count} file(s)")
    if error_count > 0:
        print(f"   Errors: {error_count} file(s)")
    print(f"   Location: {archive_base}")
    print("="*70 + "\n")


def generate_archive_summary(archives, results_dir):
    """Generate a summary file of archived logs."""
    results_path = Path(results_dir)
    archive_base = results_path / 'archive'
    summary_path = archive_base / 'ARCHIVE_SUMMARY.txt'

    with open(summary_path, 'w') as f:
        f.write("HCESVM Results Archive Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for archive_name in sorted(archives.keys()):
            files = archives[archive_name]
            f.write(f"\n{archive_name}/ ({len(files)} files)\n")
            f.write("-" * 70 + "\n")
            for file_path in sorted(files):
                filename = os.path.basename(file_path)
                f.write(f"  - {filename}\n")

    print(f"\n📝 Archive summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Archive HCESVM test result logs into dated subdirectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually execute the archival (default: dry-run mode)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be archived without actually moving files (default)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompts (use with caution!)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Path to results directory (default: auto-detect from script location)'
    )

    args = parser.parse_args()

    # Determine mode
    dry_run = not args.execute

    # Get results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Assume script is in scripts/ subdirectory
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / 'results'

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return

    print(f"\n🔍 Scanning {results_dir} for logs to archive...\n")

    # Scan for logs
    archives = scan_results_logs(results_dir)

    # Display report
    display_archive_report(archives)

    # Execute archival
    if archives:
        archive_logs(archives, results_dir, dry_run=dry_run, auto_yes=args.yes)

        # Generate summary (only if not dry-run)
        if not dry_run:
            generate_archive_summary(archives, results_dir)


if __name__ == '__main__':
    main()
