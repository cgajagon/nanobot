# Micro-Extraction Feedback Loop Research

> Date: 2026-04-04
> Status: Research complete, implementation pending
> Author: Claude Code session (systematic debugging + external research)

## Executive Summary

The micro-extractor creates a feedback loop: agent recalls facts from memory, extractor
treats them as new, duplicates accumulate. After 3 conversations, 17 events exist in the
DB with 5 near-duplicate pairs. The Jaccard dedup threshold (0.84) misses pairs like
"User's primary project is DS10540" vs "Carlos's primary project is DS10540" (Jaccard =
0.71) because entity name changes drop token overlap below threshold.

Industry frameworks (mem0, Zep, Letta) all solve this by giving the extractor knowledge
of what's already stored. The recommended fix for nanobot is **context-conditioned
extraction** (the Zep pattern): pass lightweight existing knowledge to the micro-extraction
prompt so the LLM can skip known facts.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Evidence](#evidence)
3. [Industry Survey](#industry-survey)
4. [Architecture Options](#architecture-options)
5. [Code Analysis](#code-analysis)
6. [Dedup Failure Analysis](#dedup-failure-analysis)
7. [Recommendation](#recommendation)
8. [Implementation Sketch](#implementation-sketch)

---

## Problem Statement

### The Feedback Loop

```
Turn N: User says "I'm Carlos, PM at Pratt & Whitney"
        -> Extractor saves: {fact: "Carlos is a PM at Pratt & Whitney"}

Turn N+1: System prompt includes: "Carlos is a PM at Pratt & Whitney" (from memory)
          Agent responds: "You're Carlos, a Project Manager at Pratt & Whitney..."
          -> Extractor sees this in the conversation
          -> Extractor saves: {fact: "Carlos is a PM at Pratt & Whitney"} <- DUPLICATE

Turn N+2: Same fact now has 2+ entries, higher retrieval score
          Agent more likely to mention it
          -> Extractor saves it again <- COMPOUNDING
```

### Over-Granularity

Reading a structured profile file (USER.md with 9 fields) produces 9 separate events
(one per field). While each fact is distinct, this creates retrieval noise and dedup
pressure.

---

## Evidence

### Observed Event Accumulation

After 3 conversations (greeting, "Who am I?", profile read, second "Who am I?"):

```
Total events: 17
  web,read_file: 9 events (from USER.md read)
  web: 8 events (from recall turns)

Near-duplicates (>60% Jaccard): 5 pairs
  95%: "User's primary project is DS10540..." vs "Carlos's primary project is DS10540..."
  91%: "User uses Obsidian for knowledge management." vs "Carlos uses Obsidian..."
  88%: "User speaks English and Spanish." vs "Carlos speaks English and Spanish."
  71%: "Carlos is a Project Manager at Pratt & Whitney." vs "Carlos is a PM based in Montreal."
  70%: "User's name is Carlos and is from Canada." vs "User's name is Carlos."
```

### Micro-Extraction Output Per Turn

| Turn | Events extracted | Events merged by dedup | Net new |
|------|-----------------|----------------------|---------|
| "I'm Carlos, PM at P&W" | 1 | 0 | 1 |
| Read USER.md | 9 | 0 | 9 |
| "Who am I?" (1st) | 7 | 0 | 7 |
| "Who am I?" (2nd, after prompt fix PR#144) | 5 | 2 | 3 |

---

## Industry Survey

### Framework Comparison

| Framework | Strategy | Echo Handling | Cost |
|-----------|----------|---------------|------|
| **mem0** | Extract -> vector search -> LLM classifies ADD/UPDATE/DELETE/NONE | LLM sees existing memories side-by-side with candidates | 2x LLM calls |
| **Zep** | Entity-aware extraction with existing facts in prompt | Context-conditioned extraction | 1.5x LLM calls |
| **Letta/MemGPT** | Agent self-manages via search-before-insert | Core memory visible in every prompt | 1x (but agent tool calls) |
| **LangChain** | Summary rewrite (no individual facts to dedup) | Implicit -- summary is rewritten, not appended | 1x per turn |
| **Generative Agents** | None (store everything) | Recency scoring | 1x |
| **Nanobot (current)** | Extract everything -> Jaccard/supersession dedup | Prompt instruction (PR#144) + structural dedup | 1x LLM call |

### Key Pattern: Extract-Then-Compare vs Context-Conditioned

**Pattern A: Extract-Then-Compare (nanobot's current approach)**
```
Messages -> LLM extracts all facts -> Compare each against store -> Store only novel ones
```
- Used by: mem0, classical IE systems, nanobot
- Pro: Extraction is stateless, easy to test
- Con: Wastes tokens extracting known facts; comparison may miss near-dupes

**Pattern B: Context-Conditioned Extraction (Zep pattern)**
```
Messages + existing knowledge -> LLM extracts ONLY novel facts -> Light structural dedup
```
- Used by: Zep, some RAG-enhanced systems
- Pro: Reduces extraction output to genuinely novel facts
- Con: Larger extraction prompt; retrieval may miss relevant existing memories

**Pattern C: Classify-After-Extract (mem0 pattern)**
```
Messages -> LLM extracts all -> For each, LLM classifies: ADD/UPDATE/DELETE/NONE
```
- Used by: mem0
- Pro: Most precise -- LLM sees both facts side by side
- Con: 2x+ LLM calls per turn (expensive for "best-effort" background extraction)

### Industry Convergence

The industry is converging on **"context-conditioned extraction with structural safety
net"** -- the extraction step needs to know what's already stored, with structural dedup
as a backstop. The exact implementation varies (mem0 uses separate classification, Zep
enriches the extraction prompt, Letta relies on agent self-management) but the principle
is the same.

### The Echo Problem in Literature

- **MemGPT (Packer et al., 2023, ICLR 2024)**: The LLM sees its own core memory block in
  every prompt and is instructed not to re-store known information. Relies entirely on LLM
  judgment -- no structural safety net.

- **Generative Agents (Park et al., 2023)**: No explicit dedup. Every observation stored.
  Relies on retrieval scoring to surface the most relevant/recent version. "Reflection"
  nodes consolidate over time.

- **SCM (Microsoft Research, 2023)**: Extracts structured triples, merges at triple level.
  Key insight: structural dedup is more reliable than semantic dedup for factual knowledge.

- **Self-RAG (Asai et al., 2023)**: Addresses RAG feedback loops by having the model
  critique its own retrieval results -- analogous to the echo problem.

### Best Practice: Layered Mitigation

The most robust systems combine multiple strategies:

1. **Source filtering** (cheap, first pass): Only extract from user messages
2. **Context-conditioned extraction** (medium cost): Include existing knowledge in prompt
3. **Structural dedup** (cheap, safety net): Jaccard / entity matching
4. **Temporal confirmation** (free): Re-extraction updates timestamps, not new records

---

## Architecture Options

### Option A: Context-Conditioned Extraction (Recommended)

Pass lightweight existing knowledge to the micro-extraction prompt.

**What to pass:**

| Data Source | Access Point | Token Cost |
|-------------|-------------|------------|
| Profile beliefs | `self.context.memory.profile_mgr.read_profile()` | ~100 tokens |
| Recent events | `self.context.memory.ingester.read_events(limit=15)` | ~150 tokens |
| **Total overhead** | | **~230 tokens** |

**Files changed:** `micro_extractor.py`, `message_processor.py`, `micro_extract.md`

**Pros:**
- Directly addresses feedback loop at source
- Cheap (~230 tokens, no extra LLM calls)
- Follows the Zep pattern (industry-proven)
- The LLM makes the dedup decision during extraction, not after

**Cons:**
- Profile/events must be read synchronously before async submit (~5ms)
- gpt-4o-mini may still ignore "already known" instructions sometimes
- Structural dedup still needed as safety net

### Option B: mem0-Style Classify-After-Extract

Extract all facts, then for each, search existing memories and ask LLM: ADD/UPDATE/DELETE/NONE.

**Pros:** Most precise dedup -- LLM sees both facts side by side
**Cons:** 2x+ LLM calls per turn; overkill for "best-effort" background extraction; violates
nanobot's design (micro-extraction is intentionally cheap, consolidation is authoritative)

### Option C: Improved Structural Dedup Only

Fix the Jaccard threshold and entity-awareness without changing the extraction pipeline.

**Pros:** No changes to extraction; no extra LLM calls
**Cons:** Doesn't prevent wasted extraction tokens; fragile threshold tuning; doesn't
address root cause

### Option D: Temporal Confirmation

Treat re-extraction as confirmation rather than duplication. Update `last_confirmed` and
boost confidence instead of creating new events.

**Pros:** Turns echo into a feature; no extra token cost; philosophically elegant
**Cons:** Doesn't reduce event count; conflates "agent restates" with "user confirms";
requires schema change

---

## Code Analysis

### Current Micro-Extraction Call Chain

```
message_processor.py (line 373)
  -> micro_extractor.submit(user_message, assistant_message, channel, tool_hints)
    -> _extract_and_ingest()
      -> LLM call with: system prompt (micro_extract.md) + user + assistant
      -> EventIngester.append_events()
        -> Dedup: exact ID -> supersession -> Jaccard duplicate -> new event
```

### Critical Finding: Memory Context is Discarded

```
message_processor.py line 210:
  context.build_messages() -> context.build_system_prompt()
    -> memory.get_memory_context()  <- MEMORY CONTEXT RETRIEVED HERE
    -> Rendered into system prompt string
    -> Context object DISCARDED

message_processor.py line 373:
  micro_extractor.submit()  <- MEMORY CONTEXT NO LONGER AVAILABLE
```

The memory context IS available indirectly via `self.context.memory.profile_mgr` and
`self.context.memory.ingester` -- these are accessible at the call site and don't require
re-running the full retrieval pipeline.

### Data Available at submit() Call Site (line 373)

| Data | Available? | How |
|------|-----------|-----|
| User message | Yes | `msg.content` |
| Assistant response | Yes | `final_content` |
| Channel/tools | Yes | `msg.channel`, `tool_hints` |
| System prompt string | Yes | `initial_messages[0]["content"]` |
| Memory context object | No | Discarded after rendering |
| Profile data | Yes (indirect) | `self.context.memory.profile_mgr.read_profile()` |
| Recent events | Yes (indirect) | `self.context.memory.ingester.read_events()` |
| Snapshot | Yes (indirect) | `self.context.memory.db.read_snapshot("current")` |

### Token Budget Analysis

| Component | Current | With Context |
|-----------|---------|-------------|
| System prompt (micro_extract.md) | ~200 tokens | ~430 tokens (+230 for known facts) |
| User + assistant messages | 100-300 tokens | 100-300 tokens (unchanged) |
| max_tokens for response | 500 | 500 (unchanged) |
| **Total input** | **300-500 tokens** | **530-730 tokens** |

The 230-token overhead keeps micro-extraction within gpt-4o-mini's efficient range.

---

## Dedup Failure Analysis

### Why Jaccard Misses Near-Duplicates

For "User's primary project is DS10540" vs "Carlos's primary project is DS10540":

```
Left tokens:  {user, s, ds10540, is, primary, project}     (6 tokens)
Right tokens: {carlos, s, ds10540, is, primary, project}    (6 tokens)
Overlap: {s, ds10540, is, primary, project}                  (5 tokens)
Union: {user, carlos, s, ds10540, is, primary, project}      (7 tokens)
Jaccard: 5/7 = 0.714

Duplicate threshold: 0.84 -> NOT matched
```

One token difference ("user" vs "carlos") drops similarity from ~1.0 to 0.71.

### Dedup Thresholds (from dedup.py lines 74-85)

```python
is_duplicate = (
    lexical >= 0.84                                    # high overlap
    or semantic >= 0.94                                # very high semantic
    or (lexical >= 0.6 and semantic >= 0.86)           # medium both
    or (entity_overlap >= 0.33 and (lexical >= 0.42 or semantic >= 0.52))  # entity-aware
    or (entity_overlap >= 0.30 and lexical >= 0.25 and same_type)          # entity + type
)
```

The entity-aware paths (last two conditions) would catch these duplicates, but they
require the `entities` field to be populated. The micro-extractor's tool schema includes
an `entities` field but it's not required -- gpt-4o-mini often omits it.

---

## Recommendation

### Primary: Option A (Context-Conditioned Extraction)

Pass profile beliefs + recent event summaries to the micro-extraction prompt.

### Secondary: Option C (Improved Dedup)

Lower same-type Jaccard threshold to 0.70 as a safety net.

### Combined approach (A + C):

1. Read profile + recent events at `message_processor.py` line 373
2. Pass as `known_facts` parameter to `micro_extractor.submit()`
3. Include in extraction prompt as "ALREADY KNOWN" section
4. Lower same-type Jaccard duplicate threshold from 0.84 to 0.70

**Estimated effort:** 2-3 hours including TDD
**Files changed:** 4 (`micro_extractor.py`, `message_processor.py`, `micro_extract.md`, `dedup.py`)
**No architectural boundaries crossed** -- micro-extractor already receives data from
message_processor; this adds one more parameter.

---

## Implementation Sketch

### Step 1: micro_extractor.py -- Add known_facts parameter

```python
async def submit(
    self,
    user_message: str,
    assistant_message: str,
    *,
    channel: str = "",
    tool_hints: list[str] | None = None,
    turn_timestamp: str = "",
    known_facts: str = "",           # NEW
) -> None:
```

### Step 2: message_processor.py -- Build known_facts at call site

```python
# At line 373, before submit():
known_facts = self._build_known_facts_summary()
await self._micro_extractor.submit(
    ...,
    known_facts=known_facts,
)

def _build_known_facts_summary(self) -> str:
    """Build lightweight summary of known facts for micro-extraction context."""
    lines = []
    profile = self.context.memory.profile_mgr.read_profile()
    for key in ["stable_facts", "preferences", "active_projects"]:
        for item in profile.get(key, [])[:5]:
            lines.append(f"- {item}")
    recent = self.context.memory.ingester.read_events(limit=15)
    for e in recent:
        lines.append(f"- {e.get('summary', '')}")
    return "\n".join(lines) if lines else ""
```

### Step 3: micro_extract.md -- Add ALREADY KNOWN section

```markdown
You are a memory extraction agent. ...

The following facts are ALREADY KNOWN -- do NOT extract these again:
{known_facts}

Focus on NEW information from the user's message. Skip:
- ...existing skip rules...
- Facts that match or rephrase anything in the ALREADY KNOWN list above
```

### Step 4: dedup.py -- Lower same-type threshold (safety net)

```python
# Add to the is_duplicate OR conditions:
or (lexical >= 0.70 and same_type)  # relaxed threshold for same-type events
```
