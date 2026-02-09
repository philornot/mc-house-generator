"""Auto-fix script for misnamed schematic files.

Detects actual format of files and suggests/performs corrections.
"""

import gzip
import struct
import shutil
import argparse
from pathlib import Path
from typing import Tuple


def detect_format(filepath: Path) -> str:
    """Detect actual format of schematic file.

    Args:
        filepath: Path to schematic file.

    Returns:
        Format name: "litematica", "sponge", "mcedit", or "unknown".
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            data = f.read(1000)

        # Check for format-specific tags
        if b'Regions' in data and b'BlockStates' in data:
            return "litematica"
        elif b'Palette' in data and b'BlockData' in data:
            if b'Schematic' in data[:100]:  # Check if it's in root
                return "sponge_v3"
            return "sponge"
        elif b'Blocks' in data and b'Data' in data:
            return "mcedit"
        else:
            return "unknown"

    except Exception as e:
        print(f"Error reading {filepath.name}: {e}")
        return "error"


def get_correct_extension(format_name: str) -> str:
    """Get correct file extension for format.

    Args:
        format_name: Format detected by detect_format().

    Returns:
        Correct file extension.
    """
    if format_name == "litematica":
        return ".litematic"
    elif format_name in ["sponge", "sponge_v3"]:
        return ".schem"
    elif format_name == "mcedit":
        return ".schematic"
    else:
        return None


def get_correct_directory(format_name: str, base_dir: Path = Path("houses")) -> Path:
    """Get correct directory for format.

    Args:
        format_name: Format detected by detect_format().
        base_dir: Base houses directory.

    Returns:
        Correct subdirectory path.
    """
    if format_name == "litematica":
        return base_dir / "litematic"
    elif format_name in ["sponge", "sponge_v3"]:
        return base_dir / "schem"
    elif format_name == "mcedit":
        return base_dir / "schematic"
    else:
        return None


def scan_directory(directory: Path) -> list:
    """Scan directory for misnamed files.

    Args:
        directory: Directory to scan.

    Returns:
        List of (filepath, actual_format, should_be_extension, should_be_directory).
    """
    issues = []

    for ext in [".litematic", ".schem", ".schematic"]:
        for filepath in directory.rglob(f"*{ext}"):
            actual_format = detect_format(filepath)

            if actual_format in ["litematica", "sponge", "sponge_v3", "mcedit"]:
                correct_ext = get_correct_extension(actual_format)
                correct_dir = get_correct_directory(actual_format)

                # Check if extension is wrong
                if filepath.suffix != correct_ext:
                    issues.append({
                        'filepath': filepath,
                        'current_ext': filepath.suffix,
                        'actual_format': actual_format,
                        'correct_ext': correct_ext,
                        'correct_dir': correct_dir,
                        'issue_type': 'wrong_extension'
                    })

                # Check if in wrong directory
                elif correct_dir and filepath.parent != correct_dir:
                    issues.append({
                        'filepath': filepath,
                        'current_dir': filepath.parent,
                        'actual_format': actual_format,
                        'correct_ext': correct_ext,
                        'correct_dir': correct_dir,
                        'issue_type': 'wrong_directory'
                    })

    return issues


def fix_file(issue: dict, dry_run: bool = True) -> bool:
    """Fix a misnamed/misplaced file.

    Args:
        issue: Issue dict from scan_directory().
        dry_run: If True, only show what would be done.

    Returns:
        True if fix was successful (or would be successful).
    """
    filepath = issue['filepath']
    actual_format = issue['actual_format']
    correct_ext = issue['correct_ext']
    correct_dir = issue['correct_dir']

    # Build new path
    new_name = filepath.stem + correct_ext
    new_path = correct_dir / new_name

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Fixing: {filepath.name}")
    print(f"  Format: {actual_format}")
    print(f"  Current: {filepath}")
    print(f"  New:     {new_path}")

    if not dry_run:
        try:
            # Create directory if needed
            correct_dir.mkdir(parents=True, exist_ok=True)

            # Move and rename
            shutil.move(str(filepath), str(new_path))
            print(f"  ✅ Fixed!")
            return True

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    else:
        print(f"  ℹ️  Would fix in non-dry-run mode")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Auto-detect and fix misnamed schematic files'
    )
    parser.add_argument('directory', type=str, default='houses',
                        help='Directory to scan (default: houses)')
    parser.add_argument('--fix', action='store_true',
                        help='Actually fix files (default: dry run)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all files, not just issues')

    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.exists():
        print(f"❌ Directory {directory} not found!")
        return

    print(f"Scanning {directory} for misnamed files...")
    print("=" * 70)

    issues = scan_directory(directory)

    if not issues:
        print("\n✅ All files are correctly named and placed!")
        return

    print(f"\n⚠️  Found {len(issues)} issue(s):\n")

    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['filepath'].name}")
        print(f"   Issue: {issue['issue_type']}")
        print(f"   Actual format: {issue['actual_format']}")
        if issue['issue_type'] == 'wrong_extension':
            print(f"   Current extension: {issue['current_ext']}")
            print(f"   Should be: {issue['correct_ext']}")
        else:
            print(f"   Current directory: {issue['current_dir']}")
            print(f"   Should be in: {issue['correct_dir']}")
        print()

    if args.fix:
        print("\n" + "=" * 70)
        print("APPLYING FIXES...")
        print("=" * 70)

        fixed_count = 0
        for issue in issues:
            if fix_file(issue, dry_run=False):
                fixed_count += 1

        print(f"\n✅ Fixed {fixed_count}/{len(issues)} files")

    else:
        print("\n" + "=" * 70)
        print("DRY RUN MODE (use --fix to actually apply changes)")
        print("=" * 70)

        for issue in issues:
            fix_file(issue, dry_run=True)

        print(f"\nTo fix these issues, run:")
        print(f"  python auto_fix_formats.py {directory} --fix")


if __name__ == "__main__":
    main()