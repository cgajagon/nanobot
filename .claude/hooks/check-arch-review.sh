#!/bin/bash
# Check if nanobot/ code changed but architecture docs were not updated.
# Silent on success (zero tokens). Advisory when architecture review needed.
# Worktree-aware: extracts target directory from "cd <path> &&" prefix.

INPUT=$(cat)
CMD=$(echo "$INPUT" | python -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

# Determine target directory for git commands
TARGET_DIR="."
if [[ "$CMD" =~ ^cd[[:space:]]+([^&]+)[[:space:]]*\&\& ]]; then
  TARGET_DIR="${BASH_REMATCH[1]}"
  TARGET_DIR="${TARGET_DIR%"${TARGET_DIR##*[![:space:]]}"}"
fi

# Determine what changed between current branch and origin/main
NANOBOT_CHANGED=$(git -C "$TARGET_DIR" diff origin/main...HEAD --name-only 2>/dev/null | grep "^nanobot/" | head -1)
ARCH_CHANGED=$(git -C "$TARGET_DIR" diff origin/main...HEAD --name-only 2>/dev/null | grep "^\.claude/rules/architecture\.md")
COG_CHANGED=$(git -C "$TARGET_DIR" diff origin/main...HEAD --name-only 2>/dev/null | grep "^\.claude/rules/cognitive-architecture\.md")
MEM_CHANGED=$(git -C "$TARGET_DIR" diff origin/main...HEAD --name-only 2>/dev/null | grep "^\.claude/rules/memory-architecture\.md")

# Check if memory/ code specifically changed (for memory-architecture.md)
MEMORY_CODE_CHANGED=$(git -C "$TARGET_DIR" diff origin/main...HEAD --name-only 2>/dev/null | grep "^nanobot/memory/" | head -1)

MESSAGES=""
if [ -n "$NANOBOT_CHANGED" ] && [ -z "$ARCH_CHANGED" ]; then
  MESSAGES="nanobot/ code changed but .claude/rules/architecture.md was not updated. Verify it is still accurate before pushing."
fi
if [ -n "$NANOBOT_CHANGED" ] && [ -z "$COG_CHANGED" ]; then
  if [ -n "$MESSAGES" ]; then
    MESSAGES="$MESSAGES Also: "
  fi
  MESSAGES="${MESSAGES}nanobot/ code changed but .claude/rules/cognitive-architecture.md was not updated. Verify it is still accurate before pushing."
fi
if [ -n "$MEMORY_CODE_CHANGED" ] && [ -z "$MEM_CHANGED" ]; then
  if [ -n "$MESSAGES" ]; then
    MESSAGES="$MESSAGES Also: "
  fi
  MESSAGES="${MESSAGES}nanobot/memory/ code changed but .claude/rules/memory-architecture.md was not updated. Verify it is still accurate before pushing."
fi

if [ -n "$MESSAGES" ]; then
  echo "{\"hookSpecificOutput\": {\"hookEventName\": \"PreToolUse\", \"additionalContext\": \"$MESSAGES\"}}"
fi
exit 0
