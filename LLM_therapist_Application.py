"""Backward-compatible entrypoint shim.

This module preserves legacy imports/scripts that still reference
LLM_therapist_Application.py after the main entrypoint moved to main.py.
"""

from main import app, flask_app, main, main_local, main_server


if __name__ == "__main__":
    main()
