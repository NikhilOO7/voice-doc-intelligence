#!/usr/bin/env python3
"""
Log Viewer Utility
Helps view and analyze application logs
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import re


def tail_file(filepath: Path, lines: int = 50):
    """Show last N lines of a file"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line.rstrip())


def filter_errors(filepath: Path, severity: str = "ERROR"):
    """Filter and show only error/warning messages"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    error_count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if severity in line:
                print(line.rstrip())
                error_count += 1

    print(f"\n{'='*80}")
    print(f"Total {severity} messages: {error_count}")


def filter_by_time(filepath: Path, hours: int = 1):
    """Show logs from last N hours"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    cutoff_time = datetime.now() - timedelta(hours=hours)
    cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Showing logs since: {cutoff_str}\n")

    with open(filepath, 'r') as f:
        for line in f:
            # Try to extract timestamp from log line
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                log_time_str = timestamp_match.group(1)
                log_time = datetime.strptime(log_time_str, "%Y-%m-%d %H:%M:%S")
                if log_time >= cutoff_time:
                    print(line.rstrip())
            else:
                # If no timestamp, show the line anyway (might be part of multiline log)
                print(line.rstrip())


def search_logs(filepath: Path, pattern: str, case_sensitive: bool = False):
    """Search for a pattern in logs"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_pattern = re.compile(pattern, flags)

    match_count = 0
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if compiled_pattern.search(line):
                print(f"Line {line_num}: {line.rstrip()}")
                match_count += 1

    print(f"\n{'='*80}")
    print(f"Total matches: {match_count}")


def summarize_errors(filepath: Path):
    """Summarize errors by type"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    error_types = {}
    total_errors = 0

    with open(filepath, 'r') as f:
        for line in f:
            if "ERROR" in line or "CRITICAL" in line:
                total_errors += 1
                # Try to extract error type
                error_match = re.search(r'(ERROR|CRITICAL).*?(\w+Error|\w+Exception):', line)
                if error_match:
                    error_type = error_match.group(2)
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                else:
                    error_types["Unknown"] = error_types.get("Unknown", 0) + 1

    print("Error Summary:")
    print("=" * 80)
    print(f"Total Errors: {total_errors}\n")
    print("Error Types:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="View and analyze application logs")
    parser.add_argument("--log-dir", default="logs", help="Log directory (default: logs)")
    parser.add_argument("--file", choices=["app", "error", "access"], default="app",
                       help="Which log file to view (default: app)")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Tail command
    tail_parser = subparsers.add_parser("tail", help="Show last N lines")
    tail_parser.add_argument("-n", "--lines", type=int, default=50,
                            help="Number of lines to show (default: 50)")

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter by severity")
    filter_parser.add_argument("--severity", choices=["ERROR", "WARNING", "INFO", "DEBUG"],
                              default="ERROR", help="Log severity to filter (default: ERROR)")

    # Time command
    time_parser = subparsers.add_parser("recent", help="Show logs from last N hours")
    time_parser.add_argument("--hours", type=int, default=1,
                            help="Number of hours to look back (default: 1)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for pattern")
    search_parser.add_argument("pattern", help="Pattern to search for")
    search_parser.add_argument("--case-sensitive", action="store_true",
                               help="Case sensitive search")

    # Summary command
    subparsers.add_parser("summary", help="Summarize errors")

    args = parser.parse_args()

    # Determine log file path
    log_dir = Path(args.log_dir)
    log_files = {
        "app": log_dir / "app.log",
        "error": log_dir / "error.log",
        "access": log_dir / "access.log"
    }

    filepath = log_files[args.file]

    # Execute command
    if args.command == "tail":
        tail_file(filepath, args.lines)
    elif args.command == "filter":
        filter_errors(filepath, args.severity)
    elif args.command == "recent":
        filter_by_time(filepath, args.hours)
    elif args.command == "search":
        search_logs(filepath, args.pattern, args.case_sensitive)
    elif args.command == "summary":
        if args.file == "error":
            summarize_errors(filepath)
        else:
            print("Summary command only works with --file error")
            summarize_errors(log_files["error"])
    else:
        # Default: show last 50 lines
        tail_file(filepath, 50)


if __name__ == "__main__":
    main()
