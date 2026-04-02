#!/usr/bin/env bash
# Blocks implementation commits and merges to main branch.
# Allows docs: and chore: commits on main. Silent on non-main branches.

INPUT=$(cat)
CMD=$(echo "$INPUT" | python -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

# Must contain git commit or git merge
IS_COMMIT=false
IS_MERGE=false
if echo "$CMD" | grep -q "git commit"; then
  IS_COMMIT=true
fi
if echo "$CMD" | grep -q "git merge"; then
  IS_MERGE=true
fi

if [ "$IS_COMMIT" = false ] && [ "$IS_MERGE" = false ]; then
  exit 0
fi

# Determine target directory — if command starts with "cd <path> &&", use that path
TARGET_DIR="."
if [[ "$CMD" =~ ^cd[[:space:]]+([^&]+)[[:space:]]*\&\& ]]; then
  TARGET_DIR="${BASH_REMATCH[1]}"
  # Trim trailing whitespace
  TARGET_DIR="${TARGET_DIR%"${TARGET_DIR##*[![:space:]]}"}"
fi

BRANCH=$(git -C "$TARGET_DIR" branch --show-current 2>/dev/null)

# Not on main — allow everything
if [ "$BRANCH" != "main" ]; then
  exit 0
fi

# Block merges on main — all merges should go through PRs
if [ "$IS_MERGE" = true ]; then
  echo "BLOCKED: Do not merge branches into main directly. Push the feature branch and create a PR instead." >&2
  exit 2
fi

# Allow release, docs, chore, ci commits on main
if echo "$CMD" | grep -qE '(Merge|chore|docs|ci)(\([a-zA-Z0-9_-]+\))?!?: '; then
  exit 0
fi

# Block feat, fix, refactor, test, perf on main
if echo "$CMD" | grep -qE '(feat|fix|refactor|test|perf)(\([a-zA-Z0-9_-]+\))?!?: '; then
  echo "BLOCKED: Implementation commits (feat/fix/refactor/test/perf) should not go directly to main. Create a feature branch: git worktree add ../nanobot-<branch> -b <branch>" >&2
  exit 2
fi

exit 0
