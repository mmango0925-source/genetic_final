"""Compatibility entry point for environments still using the old filename.

This module intentionally reuses the validated implementation in `main.py` so
there is only one source of truth for the symbolic-regression workflow.
"""

from main import main


if __name__ == "__main__":
    main()
