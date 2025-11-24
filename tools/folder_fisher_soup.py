#!/usr/bin/env python3
"""
Compatibility wrapper for Fisher soup helper (moved to tools/soup).
Adds project root to sys.path so running
    python tools/folder_fisher_soup.py ...
works from repository root.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.soup.folder_fisher_soup import main  # noqa: E402


if __name__ == "__main__":
    main()
