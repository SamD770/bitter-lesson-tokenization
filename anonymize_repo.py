#!/usr/bin/env python3
"""
Script to anonymize repository by removing all files except .py and .json files.
Also removes empty directories after cleanup.
"""

import os
import shutil
import argparse
from pathlib import Path


def should_keep(filepath: Path) -> bool:
    """Check if a file should be kept (is .py or .json)."""
    return filepath.suffix.lower() in {'.py', '.json'}


def get_files_to_remove(root_dir: Path) -> list[Path]:
    """Get list of all files that should be removed."""
    files_to_remove = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip .git directory
        if '.git' in dirnames:
            dirnames.remove('.git')
        
        current_dir = Path(dirpath)
        
        for filename in filenames:
            filepath = current_dir / filename
            if not should_keep(filepath):
                files_to_remove.append(filepath)
    
    return files_to_remove


def get_empty_dirs(root_dir: Path) -> list[Path]:
    """Get list of empty directories (after file removal), deepest first."""
    empty_dirs = []
    
    # Walk bottom-up to find empty directories
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Skip .git directory
        if '.git' in dirpath:
            continue
            
        current_dir = Path(dirpath)
        
        # Check if directory is empty (no files and no non-empty subdirs)
        if current_dir != root_dir:
            try:
                contents = list(current_dir.iterdir())
                if not contents:
                    empty_dirs.append(current_dir)
            except PermissionError:
                pass
    
    return empty_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Remove all files except .py and .json from repository'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--root', '-r',
        type=Path,
        default=Path(__file__).parent,
        help='Root directory to clean (default: script directory)'
    )
    args = parser.parse_args()
    
    root_dir = args.root.resolve()
    print(f"Scanning: {root_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE (will delete files)'}")
    print("-" * 60)
    
    # Get files to remove
    files_to_remove = get_files_to_remove(root_dir)
    
    if not files_to_remove:
        print("No files to remove.")
        return
    
    print(f"\nFiles to remove ({len(files_to_remove)}):")
    for f in sorted(files_to_remove):
        rel_path = f.relative_to(root_dir)
        print(f"  - {rel_path}")
    
    if not args.dry_run:
        # Actually remove files
        print("\nRemoving files...")
        for f in files_to_remove:
            try:
                f.unlink()
                print(f"  Deleted: {f.relative_to(root_dir)}")
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        
        # Remove empty directories
        print("\nRemoving empty directories...")
        # May need multiple passes as parent dirs become empty
        while True:
            empty_dirs = get_empty_dirs(root_dir)
            if not empty_dirs:
                break
            for d in empty_dirs:
                try:
                    d.rmdir()
                    print(f"  Removed empty dir: {d.relative_to(root_dir)}")
                except Exception as e:
                    print(f"  Error removing {d}: {e}")
        
        print("\nDone!")
    else:
        # Show what empty dirs would be removed
        print("\n(Dry run - checking for potentially empty directories...)")
        print("Run without --dry-run to actually delete files.")


if __name__ == '__main__':
    main()
