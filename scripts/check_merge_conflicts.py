"""Utility to detect leftover merge conflict markers in tracked files."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def iter_tracked_files() -> list[Path]:
    """Return a list of git-tracked files under the repository root."""
    result = subprocess.run(
        ["git", "ls-files"], check=True, stdout=subprocess.PIPE, text=True
    )
    root = Path.cwd()
    return [root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def file_contains_markers(path: Path) -> bool:
    """Return ``True`` if the file contains merge conflict markers."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return any(marker in text for marker in CONFLICT_MARKERS)


def main() -> int:
    self_path = Path(__file__).resolve()
    problematic = [
        path
        for path in iter_tracked_files()
        if path.resolve() != self_path and file_contains_markers(path)
    ]
    if problematic:
        print("Merge conflict markers detected in the following files:")
        for path in problematic:
            print(f" - {path}")
        return 1

    print("No merge conflict markers detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
