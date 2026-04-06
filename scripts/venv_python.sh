#!/usr/bin/env bash
# Find the project's venv Python, falling back to system python.
# Used by pre-commit hooks to ensure the correct Python is used
# even when a different system Python is on PATH.
#
# Searches for .venv in the current directory first, then in the
# git toplevel (handles worktrees where .venv is at the main repo root).

_find_venv_python() {
    local root
    root="$(git rev-parse --show-toplevel 2>/dev/null)"

    if [ -n "$VIRTUAL_ENV" ]; then
        echo "python"
        return
    fi

    # Check current directory
    if [ -f ".venv/Scripts/python.exe" ]; then
        echo ".venv/Scripts/python.exe"
        return
    fi
    if [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
        return
    fi

    # Check git toplevel (for worktrees)
    if [ -n "$root" ]; then
        if [ -f "$root/.venv/Scripts/python.exe" ]; then
            echo "$root/.venv/Scripts/python.exe"
            return
        fi
        if [ -f "$root/.venv/bin/python" ]; then
            echo "$root/.venv/bin/python"
            return
        fi
    fi

    echo "python"
}

exec "$(_find_venv_python)" "$@"
