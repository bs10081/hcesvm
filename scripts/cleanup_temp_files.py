#!/usr/bin/env python3
"""
HCESVM Project Cleanup Script

Identifies and removes temporary files from the project root directory.

Usage:
    python scripts/cleanup_temp_files.py [--dry-run] [--yes]

Options:
    --dry-run    Show what would be deleted without actually deleting (default)
    --execute    Actually execute the cleanup
    --yes        Skip confirmation prompts (use with caution!)

Examples:
    # Preview what would be deleted
    python scripts/cleanup_temp_files.py

    # Execute cleanup with confirmation
    python scripts/cleanup_temp_files.py --execute

    # Execute cleanup without confirmation (dangerous!)
    python scripts/cleanup_temp_files.py --execute --yes
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime


# Files to always keep in root directory
KEEP_FILES = {
    'README.md',
    'CLAUDE.md',
    'FINAL_TEST_RESULTS.md',
    'CLASS1_FIRST_SUMMARY.md',
    'pyproject.toml',
    'uv.lock',
    '.gitignore'
}

# Patterns for temporary files to delete
TEMP_PATTERNS = [
    '*.log',           # Temporary log files
    'monitor_*.sh',    # Monitoring scripts
    'test_*.py',       # Temporary test scripts
    'generate_*.py',   # Temporary generation scripts
    '*.backup',        # Backup files
    '*.bak'            # Backup files
]


def get_file_info(filepath):
    """Get file size and modification time."""
    stat = os.stat(filepath)
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    return size, mtime


def format_size(size_bytes):
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def scan_temp_files(root_dir):
    """
    Scan for temporary files in root directory.

    Returns:
        list: List of file paths to delete
    """
    root_path = Path(root_dir)
    delete_files = []

    for pattern in TEMP_PATTERNS:
        matches = glob.glob(str(root_path / pattern))
        for file_path in matches:
            basename = os.path.basename(file_path)
            # Skip files in KEEP_FILES
            if basename not in KEEP_FILES:
                delete_files.append(file_path)

    return sorted(delete_files)


def display_cleanup_report(files):
    """Display a formatted report of files to be cleaned."""
    if not files:
        print("✅ No temporary files found. Project is clean!")
        return

    total_size = sum(os.path.getsize(f) for f in files)

    print("\n" + "="*70)
    print(" 🗑️  CLEANUP RECOMMENDATIONS")
    print("="*70)
    print(f"\nFound {len(files)} temporary file(s) - Total size: {format_size(total_size)}\n")

    # Group by pattern
    by_pattern = {}
    for file in files:
        basename = os.path.basename(file)
        if basename.endswith('.log'):
            pattern = '*.log'
        elif basename.startswith('monitor_'):
            pattern = 'monitor_*.sh'
        elif basename.startswith('test_'):
            pattern = 'test_*.py'
        elif basename.startswith('generate_'):
            pattern = 'generate_*.py'
        elif basename.endswith('.backup'):
            pattern = '*.backup'
        elif basename.endswith('.bak'):
            pattern = '*.bak'
        else:
            pattern = 'other'

        if pattern not in by_pattern:
            by_pattern[pattern] = []
        by_pattern[pattern].append(file)

    # Display by pattern
    for pattern, pattern_files in sorted(by_pattern.items()):
        print(f"  [{pattern}] {len(pattern_files)} file(s)")
        for file_path in pattern_files:
            size, mtime = get_file_info(file_path)
            basename = os.path.basename(file_path)
            print(f"    ❌ {basename:40s} {format_size(size):>10s}  (modified: {mtime})")
        print()

    print("="*70)


def cleanup_files(files, dry_run=True, auto_yes=False):
    """
    Delete files after confirmation.

    Args:
        files: List of file paths to delete
        dry_run: If True, only show what would be deleted
        auto_yes: If True, skip confirmation prompt
    """
    if not files:
        print("✅ No files to clean up.")
        return

    if dry_run:
        print("\n[DRY RUN MODE] No files were actually deleted.")
        print("Run with --execute flag to perform actual cleanup.\n")
        return

    # Confirmation
    if not auto_yes:
        print("\n⚠️  WARNING: This will permanently delete the files listed above!")
        response = input("Proceed with cleanup? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cleanup cancelled by user.")
            return

    # Execute deletion
    print("\n🗑️  Executing cleanup...\n")
    success_count = 0
    error_count = 0

    for file_path in files:
        try:
            os.remove(file_path)
            print(f"  ✅ Deleted: {os.path.basename(file_path)}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error deleting {os.path.basename(file_path)}: {e}")
            error_count += 1

    print("\n" + "="*70)
    print(f"✅ Cleanup complete!")
    print(f"   Deleted: {success_count} file(s)")
    if error_count > 0:
        print(f"   Errors: {error_count} file(s)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Clean up temporary files from HCESVM project root directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually execute the cleanup (default: dry-run mode)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be deleted without actually deleting (default)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompts (use with caution!)'
    )

    args = parser.parse_args()

    # Determine mode
    dry_run = not args.execute

    # Get project root (assume script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"\n🔍 Scanning {project_root} for temporary files...\n")

    # Scan for temporary files
    temp_files = scan_temp_files(project_root)

    # Display report
    display_cleanup_report(temp_files)

    # Execute cleanup
    if temp_files:
        cleanup_files(temp_files, dry_run=dry_run, auto_yes=args.yes)


if __name__ == '__main__':
    main()
