#!/usr/bin/env bash
# Find the project's venv Python, falling back to system python.
# Used by pre-commit hooks to ensure the correct Python is used
# even when a different system Python is on PATH.
#
# Searches for .venv in the current directory, git toplevel, and
# the main checkout root (via --git-common-dir, for worktrees).

_find_venv_python() {
    local toplevel common_root

    if [ -n "$VIRTUAL_ENV" ]; then
        echo "python"
        return
    fi

    toplevel="$(git rev-parse --show-toplevel 2>/dev/null)"

    # For worktrees: --git-common-dir points to the main repo's .git,
    # so the main checkout root is its parent directory.
    common_root="$(git rev-parse --git-common-dir 2>/dev/null)"
    if [ -n "$common_root" ]; then
        common_root="$(cd "$common_root/.." 2>/dev/null && pwd)"
    fi

    # Check paths in priority order
    for dir in "." "$toplevel" "$common_root"; do
        [ -z "$dir" ] && continue
        if [ -f "$dir/.venv/Scripts/python.exe" ]; then
            echo "$dir/.venv/Scripts/python.exe"
            return
        fi
        if [ -f "$dir/.venv/bin/python" ]; then
            echo "$dir/.venv/bin/python"
            return
        fi
    done

    echo "python"
}

exec "$(_find_venv_python)" "$@"
