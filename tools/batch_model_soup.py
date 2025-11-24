#!/usr/bin/env python3
"""
Compatibility wrapper for soup utilities now located under tools/soup.
Ensures project root is on sys.path so it works when executed as:
    python tools/batch_model_soup.py ...
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.soup.batch_model_soup import main  # noqa: E402


if __name__ == "__main__":
    main()
