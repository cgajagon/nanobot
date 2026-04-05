# Architecture Review: Passing Memory Context to the Micro-Extractor

> Date: 2026-04-04
> Status: Complete
> Scope: Rigorous evaluation of whether memory context should be passed into the
> micro-extraction pipeline
> Sources: 3 deep-dive research agents (code audit, boundary analysis, external research),
> plus earlier session research

---

## 1. Executive Summary

**Proposal under review:** Pass the memory context (profile beliefs, recent events) to
the micro-extractor so it can compare newly extracted information against existing
knowledge before deciding what to extract.

**Verdict: Do not implement as proposed.**

The evidence from five production memory frameworks (mem0, Zep/Graphiti, Letta, cognee),
academic systems (NELL), and empirical research on LLM context degradation uniformly
points in the same direction: **extraction and reconciliation should remain separate
concerns**. Every production system that has achieved scale uses a pipeline where
extraction is mostly stateless and comparison/dedup is a separate downstream step.

The proposed change would:
- Violate the codebase's established import boundaries (`agent/` accessing `memory/read/`)
- Couple the write path to the read path's schema
- Degrade extraction quality (empirically demonstrated by context rot research)
- Contradict the Single Responsibility Principle
- Add complexity for marginal benefit over improving the existing dedup pipeline

**Recommended alternative:** Strengthen the existing dedup pipeline in `EventIngester`
(Option C in Section 10), which addresses the same problem without architectural cost.

---

## 2. Current Architecture Overview

### Memory Write Pipeline

```
                    MICRO-EXTRACTION PATH              CONSOLIDATION PATH
                    (per-turn, background)             (periodic, authoritative)
                           |                                    |
                    MicroExtractor                     ConsolidationPipeline
                    (gpt-4o-mini, 500 tokens)          (main model, full context)
                           |                                    |
                    Receives: user_msg +               Receives: conversation +
                    assistant_msg ONLY                  memory snapshot + profile
                           |                                    |
                    MemoryEvent.from_dict()             EventCoercer.coerce_event()
                           |                                    |
                           +------- CONVERGENCE POINT ----------+
                           |
                    EventIngester.append_events()
                           |
                    ┌──────┴───────────────────────┐
                    │  1. Exact ID dedup (PK)       │
                    │  2. Supersession (FTS5+neg)   │
                    │  3. Semantic duplicate (FTS5   │
                    │     + Jaccard >= 0.84)         │
                    │  4. New event → write          │
                    └──────────────────────────────┘
```

### Key Design Properties

- **MicroExtractor is decoupled from the read path.** It has no access to
  `MemoryRetriever`, `ProfileStore.read_profile()`, `ContextAssembler`, or any
  read-side component. Its only dependencies are `LLMProvider` (generic),
  `EventIngester` (write-side), and `Embedder` (protocol).

- **The "should this fact be stored?" decision is distributed.** MicroExtractor
  asks "is this extraction-worthy?" (based on the turn). EventIngester asks "is
  this storage-worthy?" (based on existing events). These are separate concerns
  that change for different reasons.

- **Micro-extraction is explicitly best-effort.** The design spec states: "If rapid
  successive turns both trigger extraction, both background tasks run concurrently.
  `append_events` deduplication reads existing events at the start of each call...
  This is acceptable — full consolidation's supersession logic will clean up any
  duplicates."

- **Consolidation IS context-aware by design.** The full consolidation pipeline
  receives the memory snapshot and current profile. It is the authoritative memory
  write path. Micro-extraction is a lightweight optimization, not a replacement.

### Data Available at the Micro-Extraction Call Site

At `message_processor.py` line 373:

| Data | Available? | How |
|------|-----------|-----|
| User message | Yes | `msg.content` |
| Assistant response | Yes | `final_content` |
| Channel/tools | Yes | `msg.channel`, `tool_hints` |
| System prompt string | Yes | `initial_messages[0]["content"]` |
| Memory context object | **No** | Discarded after rendering into system prompt |
| Profile data | Yes (indirect) | `self.context.memory.profile_mgr.read_profile()` |
| Recent events | Yes (indirect) | `self.context.memory.ingester.read_events()` |

The memory context is built in `ContextBuilder.build_system_prompt()` at line 159 of
`context.py`, rendered into the system prompt string, then discarded. By the time
micro-extraction runs, the structured memory context no longer exists as a data object.

---

## 3. Current Weaknesses and Pain Points

### 3.1 The Feedback Loop (Critical)

When the agent recalls facts from memory and states them in its response, the
micro-extractor treats the response as new information:

```
Turn 1: User says "I'm Carlos, PM at Pratt & Whitney"
        → Extractor saves: "Carlos is a PM at Pratt & Whitney"

Turn 2: Agent responds: "You're Carlos, a Project Manager at Pratt & Whitney..."
        → Extractor saves: "Carlos is a PM at Pratt & Whitney" ← DUPLICATE
```

After 3 conversations: 17 events, 5 near-duplicate pairs.

### 3.2 Entity Name Mismatch in Dedup (High)

The Jaccard dedup threshold (0.84) misses pairs with entity name changes:

```
"User's primary project is DS10540"  vs  "Carlos's primary project is DS10540"

Tokens: {user,s,ds10540,is,primary,project} vs {carlos,s,ds10540,is,primary,project}
Jaccard: 5/7 = 0.714 → BELOW 0.84 threshold → NOT matched as duplicate
```

One token difference drops similarity from ~1.0 to 0.71. The entity-aware dedup paths
(last two conditions in the threshold check) require populated `entities` fields, which
gpt-4o-mini often omits.

### 3.3 Semantic Similarity is a Stub (Medium)

In `dedup.py` line 48: `semantic = lexical`. The semantic similarity computation always
returns the same value as lexical (Jaccard). No embedding-based similarity is computed.
This means:
- The condition `semantic >= 0.94` is equivalent to `lexical >= 0.94`
- The condition `lexical >= 0.6 and semantic >= 0.86` effectively requires `lexical >= 0.86`
- Several dedup thresholds are unreachable

### 3.4 Over-Granular Extraction (Low)

Reading a structured file (USER.md with 9 fields) produces 9 separate events. The
micro-extraction prompt (updated in PR#144) now guides toward 1-3 events per turn,
but gpt-4o-mini partially ignores this at temperature=0.0.

---

## 4. Detailed Analysis of the Proposed Refactor

### What the Proposal Entails

1. `message_processor.py` reads profile beliefs + recent events (~230 tokens)
2. Passes them as a `known_facts` parameter to `MicroExtractor.submit()`
3. `MicroExtractor._extract_and_ingest()` includes known facts in the system prompt
4. The extraction LLM (gpt-4o-mini) sees existing knowledge and skips known facts

### Intended Benefits

- **Feedback loop prevention:** The LLM sees what's already stored and avoids
  re-extracting it
- **Reduced dedup pressure:** Fewer events reach the ingestion pipeline
- **Token efficiency:** Fewer wasted extraction tokens on known facts

### Actual Implications

**Architecture:** Creates a new dependency from the agent layer (`message_processor.py`)
to the memory read path (`profile_mgr.read_profile()`, `ingester.read_events()`). The
current architecture enforces that `agent/*` does not runtime-import from `memory/read/`.

**Data flow:** The micro-extractor, currently a pure write-path component, gains
implicit knowledge of the read-path schema (profile structure, event format, metadata
fields).

**Coupling:** Changes to `ProfileStore` schema, `EventStore` format, or
`ContextAssembler` enrichment would now affect MicroExtractor — a component that
currently has zero read-path dependencies.

**Performance:** Adds ~5ms synchronous read (profile + events) before the async submit.
Adds ~230 tokens to the extraction prompt, increasing gpt-4o-mini cost per turn.

**Correctness:** Introduces a race condition window — profile may change between read
and extraction completion (since extraction is async/background).

**Testability:** Currently, MicroExtractor tests are trivial (mock provider + mock
ingester, no database). Adding known_facts requires tests that understand profile
schema, event format, and enrichment behavior.

---

## 5. Architectural Pros

1. **Directly addresses the feedback loop** at the source rather than downstream
2. **Follows the "prompt is smart" pattern** — the LLM makes the decision, guided
   by context in the prompt
3. **Low token overhead** (~230 tokens) — well within gpt-4o-mini's capacity
4. **No new LLM calls** — reuses the existing extraction call with a richer prompt
5. **Conceptually simple** — "tell the extractor what you already know" is intuitive

---

## 6. Architectural Cons

1. **Violates import boundaries.** `agent/message_processor.py` must access
   `memory.read` internals (profile_mgr, ingester.read_events). Architecture rules
   explicitly forbid `agent/*` from runtime-importing `memory` internals.

2. **Couples write path to read path.** MicroExtractor (write-side) gains implicit
   dependency on ProfileStore and EventStore schema (read-side). These currently
   evolve independently.

3. **Breaks structural decoupling.** Currently, MicroExtractor *cannot* see the read
   path — the decoupling is enforced by the lack of access. With context passing,
   decoupling becomes a convention, not a guarantee.

4. **Violates Single Responsibility.** "Extract facts from this turn" becomes "extract
   facts from this turn that are not already known." These change for different reasons:
   extraction logic changes when conversation format or LLM changes; comparison logic
   changes when dedup thresholds or profile schema changes.

5. **Contradicts the "best-effort" contract.** Micro-extraction is explicitly designed
   as a lightweight optimization where "full consolidation remains the authoritative
   memory pipeline." Adding comparison logic elevates it toward an authoritative role
   without the robustness guarantees.

6. **Degrades extraction quality (empirically).** Context rot research (Chroma, 2025)
   shows "significant degradation begins at ~500-750 tokens for semantic tasks" and
   "when information semantically matches surrounding content, extraction becomes
   harder." Passing existing memory events creates exactly this condition.

7. **Increases testing burden.** Unit tests must now mock profile schema, event format,
   and handle synchronization scenarios.

---

## 7. Risks and Failure Modes

### 7.1 Context Pollution Degrades Extraction

**Risk:** The extraction LLM becomes biased toward confirming existing beliefs rather
than extracting genuinely new contradicting facts.

**Evidence:** The "Lost in the Middle" study (Liu et al., 2024) showed 30%+ performance
degradation when relevant information shifts to the middle of context. Existing memories
push extraction targets (the actual conversation) toward the middle.

**Severity:** HIGH — this directly undermines the extraction pipeline's purpose.

### 7.2 Schema Coupling Creates Fragility

**Risk:** A change to `ProfileStore` (adding a field, renaming a key, changing the
belief lifecycle) silently breaks the extraction prompt formatting.

**Severity:** MEDIUM — would manifest as degraded extraction quality, not a crash.

### 7.3 Race Condition Between Read and Write

**Risk:** Profile is read at submit time, but extraction runs asynchronously. Between
read and extraction, another turn may update the profile (via live correction or
consolidation). The extraction LLM operates on stale context.

**Severity:** LOW — worst case is a duplicate that dedup catches.

### 7.4 Stale or Incorrect Memories Harm Extraction

**Risk:** If existing memories contain errors (from previous extraction mistakes),
passing them to the extractor reinforces those errors. The extractor may avoid
extracting a correct new fact because an incorrect existing fact looks similar.

**Evidence:** Research (arXiv:2404.12957) shows "incorrect examples are more harmful
than unknown ones" — wrong context actively degrades model performance.

**Severity:** MEDIUM — compounds over time.

### 7.5 Architectural Drift

**Risk:** Once the boundary between extraction and reconciliation is blurred, future
sessions may add more read-path dependencies to the extractor (graph entities,
retrieval scores, strategy data), gradually turning it into a coupled monolith.

**Severity:** HIGH for long-term maintainability.

---

## 8. External Best Practices and Research Findings

### Production Framework Consensus

| Framework | Extraction receives existing knowledge? | Reconciliation separate? |
|-----------|----------------------------------------|--------------------------|
| **mem0** | No (extraction phase) / Yes (update phase) | Yes — two distinct LLM calls |
| **Graphiti/Zep** | No (just recent 3 episodes for pronouns) | Yes — three phases |
| **Letta/MemGPT** | N/A (agent self-manages in-place) | No separation (bounded memory only) |
| **cognee** | No | Yes — separate pipeline stage |
| **NELL** | No (extractors are stateless) | Yes — Knowledge Integrator |

**Every production system with scale separates extraction from reconciliation.**
Extractors are kept stateless or near-stateless. Comparison happens downstream.

### mem0's Architecture (Most Directly Relevant)

mem0's `add()` pipeline runs two phases:
1. **Extraction:** LLM receives conversation only. No existing memories.
2. **Reconciliation:** For each extracted fact, vector search retrieves top-10 similar
   existing memories. A second LLM call classifies: ADD / UPDATE / DELETE / NOOP.

The reconciliation prompt is separately customizable. This explicit separation allows
modifying reconciliation logic without touching extraction.

### Graphiti's Three-Phase Design (Most Rigorous)

Graphiti (Zep's open-source temporal knowledge graph) implements:
1. **Extraction:** `extract_nodes()` and `extract_edges()` receive only episode content
   + entity type mappings + previous 3 episodes (for pronoun resolution). The prompt
   explicitly states: "Exclude entities mentioned only in the PREVIOUS MESSAGES."
2. **Resolution:** `resolve_extracted_nodes()` uses exact match → fuzzy similarity →
   LLM reasoning. Has full access to the existing graph.
3. **Temporal Reconciliation:** `resolve_edge_contradictions()` handles fact
   invalidation with bitemporal tracking.

Quote from Graphiti's maintainers: "Each extraction/dedup task has its own separate
prompt, which makes output faster, more accurate, and easier to test, and allows many
tasks to run in parallel."

### NELL's Knowledge Integrator (Academic Validation)

NELL ran 24/7 from 2010 building 120M+ beliefs. Multiple independent extractors
operated on web text only — NO access to the current KB. The Knowledge Integrator
reconciled all candidates against the KB using confidence scoring and consistency
constraints. This is the canonical example of separated extraction and reconciliation.

### Empirical Evidence Against Context-Conditioned Extraction

| Finding | Source | Implication |
|---------|--------|-------------|
| Degradation at 500-750 tokens for semantic tasks | Context Rot (Chroma, 2025) | 230 tokens of existing knowledge approaches the degradation zone |
| Semantic blending makes extraction harder | Context Rot (Chroma, 2025) | Passing related memories creates exactly this condition |
| 30%+ degradation when targets in middle of context | Lost in the Middle (Liu et al., 2024) | Existing memories push extraction targets to middle |
| Length hurts even with perfect retrieval | EMNLP 2025 (arXiv:2510.05381) | More context = worse reasoning, regardless of findability |
| Incorrect examples more harmful than unknown | arXiv:2404.12957 | Stale/wrong memories actively harm extraction |
| Generic prompt additions degrade accuracy | arXiv:2601.22025 | "Already known" instructions may conflict with extraction objective |

### CQRS / Event Sourcing Parallel

The CQRS pattern provides a direct architectural analogy:
- **Write side (extraction):** Append-only, stateless. Events captured as-is.
- **Read side (reconciliation):** Projections built from events, can be rebuilt
  independently.

The write path is deliberately kept simple. Reconciliation happens asynchronously.
This maps directly to nanobot's `MicroExtractor` → `EventIngester` pipeline.

---

## 9. Better Alternative Designs

### Alternative A: Enhanced Structural Dedup (Recommended)

**Core idea:** Improve the existing `EventIngester` dedup pipeline to catch the
near-duplicates it currently misses, without changing the extraction pipeline at all.

**Changes:**

1. **Lower same-type Jaccard threshold** from 0.84 to 0.70 when events share the
   same type. This catches "User's primary project..." vs "Carlos's primary project..."
   (Jaccard = 0.71).

2. **Add entity name normalization** before Jaccard computation. Treat "User", "user",
   and the configured user name (from profile or USER.md) as equivalent tokens.

3. **Implement embedding-based semantic similarity** instead of the current stub
   (`semantic = lexical`). The embedder is already available in the ingester
   constructor — use it for a real cosine similarity check.

**Advantages:**
- Zero changes to MicroExtractor or MessageProcessor
- No new coupling or boundary violations
- Changes isolated to `dedup.py` and `ingester.py` (write-side only)
- Independently testable with synthetic events
- Catches duplicates from ALL sources (micro-extraction AND consolidation)

**Disadvantages:**
- Lower threshold risks false-positive merges for genuinely different facts
- Entity normalization requires knowing the user's name (config dependency)
- Embedding similarity adds latency to ingestion (~50ms per event)

**Complexity:** LOW — changes to 2 files, ~50 lines of code
**Migration effort:** None — backward compatible
**Suitability:** HIGH — works within existing architecture

### Alternative B: Post-Extraction Reconciliation Layer

**Core idea:** Add a new `ExtractionReconciler` class in `memory/write/` that sits
between extraction and ingestion. It receives extracted events + existing knowledge
and filters out known facts before passing to the ingester.

```
MicroExtractor → ExtractionReconciler → EventIngester
                      ↑
              ProfileStore.read_profile()
              Ingester.read_events(limit=15)
```

**Advantages:**
- Keeps extraction stateless (MicroExtractor unchanged)
- Reconciliation is a separate, testable component
- Lives in `memory/write/` — no boundary violations
- Single responsibility: reconcile extracted events against known state

**Disadvantages:**
- Adds a new component and file to `memory/write/` (currently 10 files)
- Still requires reading profile/events at extraction time (latency)
- Adds another layer to the pipeline (extraction → reconciliation → ingestion)
- The reconciler needs to understand profile schema (coupling, just in a different place)

**Complexity:** MEDIUM — new class, new tests, wiring in factory
**Migration effort:** LOW — additive, no existing code changes
**Suitability:** MEDIUM — architecturally clean but adds complexity for marginal gain

### Alternative C: Temporal Confirmation (Reframe the Problem)

**Core idea:** Instead of preventing duplicate extraction, treat re-extraction as
confirmation. When a near-duplicate is detected (Jaccard 0.60-0.84), update the
existing event's `last_confirmed` timestamp and boost confidence rather than
creating a new record or merging.

```python
# In EventIngester.append_events(), between supersession and duplicate checks:
if 0.60 <= jaccard < 0.84 and same_type:
    existing["last_confirmed"] = now_iso
    existing["confidence"] = min(1.0, existing["confidence"] + 0.03)
    _write_events([existing], embeddings=None)
    confirmed += 1
    continue  # Skip to next candidate
```

**Advantages:**
- Turns the feedback loop from a bug into a feature
- Frequently mentioned facts gain higher confidence (useful signal)
- No changes to extraction pipeline
- No new coupling
- No extra token cost

**Disadvantages:**
- Doesn't reduce event count (events still extracted, just not duplicated)
- Conflates "agent restates" with "user confirms" — different signals
- Requires schema evolution (add `last_confirmed` field)
- Retrieval scoring must factor in `last_confirmed` for the signal to matter

**Complexity:** LOW-MEDIUM — schema change + scoring adjustment
**Migration effort:** LOW — additive field, backward compatible
**Suitability:** MEDIUM — elegant but solves a slightly different problem

### Alternative D: Source-Aware Extraction (Prompt-Only)

**Core idea:** Instead of passing existing knowledge to the extractor, improve the
prompt to distinguish user-originated facts from agent-recalled facts based on
message roles alone.

The current micro-extraction prompt receives `[system, user_message, assistant_message]`.
The assistant message contains both new information AND recalled facts. Instruct the LLM
to only extract from the user message, treating the assistant message as context only.

```markdown
## Extraction Rules
- Extract NEW facts from the USER message only
- The ASSISTANT message is provided for context only — it may contain
  facts recalled from memory that should NOT be re-extracted
- If the assistant is summarizing what it knows about the user, those
  facts are already stored — skip them entirely
```

**Advantages:**
- Zero code changes — prompt-only
- No new coupling, no boundary violations
- No extra tokens (actually fewer, since we're restricting, not adding)
- Already partially implemented in PR#144

**Disadvantages:**
- gpt-4o-mini may not reliably follow this instruction (observed in PR#144)
- Can't distinguish "user confirms existing fact" from "user states new fact"
- Doesn't help when the user provides new information AND the assistant restates
  existing facts in the same turn

**Complexity:** ZERO — prompt change only
**Migration effort:** ZERO
**Suitability:** LOW-MEDIUM — correct direction but unreliable with cheap models

---

## 10. Recommended Target Architecture

### Primary: Alternative A (Enhanced Structural Dedup)

This is the correct fix because:

1. **It addresses the actual failure point.** The problem is not that extraction
   produces duplicates — it's that dedup doesn't catch them. The Jaccard threshold
   of 0.84 is too high, and the semantic similarity is a stub.

2. **It respects existing architecture.** No boundary violations, no new coupling,
   no changes to the extraction pipeline. Changes are isolated to the write-side
   dedup layer where they belong.

3. **It's universally effective.** Improved dedup catches duplicates from ALL
   sources — micro-extraction, consolidation, and any future write path.

4. **It's supported by the industry pattern.** Every production framework
   (mem0, Graphiti, cognee) uses downstream reconciliation, not context-conditioned
   extraction.

### Secondary: Alternative D (Source-Aware Prompt) + Alternative C (Temporal Confirmation)

As defense-in-depth layers:
- The prompt improvement (already in PR#144) reduces extraction of recalled facts
- Temporal confirmation turns remaining duplicates into useful confidence signals

### Architecture Principle to Document

Add to `memory-architecture.md`:

> **Extraction is stateless, reconciliation is write-time.** The micro-extractor
> receives conversation content only. It does NOT receive existing events, profile
> beliefs, or prior extractions. All comparison, dedup, supersession, and merge logic
> lives in `EventIngester.append_events()`. This separation exists because:
> (1) empirical evidence shows context pollution degrades extraction accuracy,
> (2) extraction and reconciliation change for different reasons (SRP),
> (3) stateless extraction is independently testable, and
> (4) production memory systems universally separate these concerns.

---

## 11. Migration Strategy

### Phase 1: Enhanced Dedup (Immediate)

1. Lower same-type Jaccard threshold from 0.84 to 0.70 in `dedup.py`
2. Add entity name normalization (treat "User" and configured user name as equivalent)
3. Tests: synthetic event pairs that currently slip through dedup

### Phase 2: Temporal Confirmation (Near-term)

1. Add `last_confirmed` field to MemoryEvent
2. When near-duplicate detected (0.60 <= Jaccard < new_threshold), update existing
   event's `last_confirmed` and confidence instead of creating/merging
3. Factor `last_confirmed` into retrieval scoring

### Phase 3: Real Semantic Similarity (Future)

1. Replace `semantic = lexical` stub with actual embedding cosine similarity
2. Use the embedder already available in the ingester
3. Adjust thresholds based on empirical testing

### What NOT to Do

- Do NOT pass memory context to MicroExtractor
- Do NOT add a reconciliation layer between extraction and ingestion (unnecessary
  complexity given improved dedup)
- Do NOT increase micro-extraction's `max_tokens` budget
- Do NOT make micro-extraction synchronous (it must remain background/best-effort)

---

## 12. Implementation Considerations

### Dedup Threshold Tuning

Lowering from 0.84 to 0.70 requires careful testing:
- **True positives:** "User's project is DS10540" vs "Carlos's project is DS10540"
  (Jaccard 0.71) — should merge
- **False positive risk:** "User prefers Python for data science" vs "User prefers
  Python for web development" (may have Jaccard > 0.70) — should NOT merge

Mitigation: Only apply the lower threshold when `candidate_type == existing_type`
(same event type). This is already a condition in the dedup logic.

### Entity Name Normalization

The user's name is available from:
- `ProfileStore.read_profile()["stable_facts"]` — but reading profile in the
  ingester crosses a boundary
- `USER.md` bootstrap file — but reading files in the ingester is wrong
- A configured `user_name` constant — simplest, passed at construction time

Recommendation: Add `user_aliases: set[str]` parameter to `EventDeduplicator.__init__()`,
populated from config in the factory. The dedup logic normalizes these aliases to a
canonical token before computing Jaccard.

### Backward Compatibility

All changes are additive:
- Lower threshold catches more duplicates (merges instead of creating)
- `last_confirmed` field defaults to `None` for existing events
- Real semantic similarity is a drop-in replacement for the stub

---

## 13. Testing and Validation Strategy

### Unit Tests (dedup.py)

```python
# Test the specific failure case
def test_entity_name_mismatch_detected_as_duplicate():
    """'User's project is DS10540' vs 'Carlos's project is DS10540' should merge."""
    dedup = EventDeduplicator(user_aliases={"user", "carlos"})
    left = {"type": "fact", "summary": "User's primary project is DS10540"}
    right = {"type": "fact", "summary": "Carlos's primary project is DS10540"}
    idx, score = dedup.find_semantic_duplicate(right, [left])
    assert idx == 0  # Should match

def test_different_facts_not_merged():
    """Similar structure but different content should not merge."""
    dedup = EventDeduplicator()
    left = {"type": "fact", "summary": "User prefers Python for data science"}
    right = {"type": "fact", "summary": "User prefers Java for web development"}
    idx, score = dedup.find_semantic_duplicate(right, [left])
    assert idx is None  # Should NOT match

def test_lowered_threshold_same_type_only():
    """Lower threshold only applies when event types match."""
    dedup = EventDeduplicator()
    left = {"type": "fact", "summary": "User works on DS10540"}
    right = {"type": "task", "summary": "User works on DS10540"}
    idx, score = dedup.find_semantic_duplicate(right, [left])
    assert idx is None  # Different types, higher threshold applies
```

### Contract Tests

```python
def test_feedback_loop_events_merged():
    """Events from agent recall should merge with existing facts."""
    store = MemoryStore(tmp_path)
    # Simulate: user states fact, then agent recalls it
    event1 = MemoryEvent(type="fact", summary="User's project is DS10540")
    event2 = MemoryEvent(type="fact", summary="Carlos's project is DS10540")
    store.ingester.append_events([event1])
    store.ingester.append_events([event2])
    events = store.ingester.read_events()
    # Should be 1 event (merged), not 2
    fact_events = [e for e in events if "DS10540" in e.get("summary", "")]
    assert len(fact_events) == 1
```

### Integration Test

```python
@pytest.mark.llm
async def test_micro_extraction_feedback_loop_controlled():
    """After micro-extraction + dedup, no duplicate facts accumulate."""
    # Setup: pre-populate memory with known facts
    # Run: agent recalls facts in response
    # Assert: event count doesn't grow from recall turns
```

---

## 14. Open Questions

1. **What is the optimal Jaccard threshold?** 0.70 is a hypothesis based on the
   "User" vs "Carlos" case (0.71). Empirical testing with real event pairs is needed
   to find the sweet spot between catching duplicates and avoiding false merges.

2. **Should the semantic similarity stub be fixed now or later?** Real embedding
   cosine similarity would make the dedup pipeline significantly more robust, but
   adds ~50ms latency per event and requires the embedder to be available.

3. **Should entity name normalization use a fixed alias set or dynamic resolution?**
   Fixed aliases (from config) are simple but brittle. Dynamic resolution (from the
   knowledge graph's entity linker) is more robust but adds coupling.

4. **Is the consolidation window of 100 messages too high?** With micro-extraction
   handling per-turn events, the consolidation pipeline's role shifts toward
   profile updates and snapshot rebuilds. A lower window might improve profile
   accuracy without excessive LLM cost.

5. **Should the micro-extraction prompt restrict extraction to user messages only?**
   PR#144 added guidance but gpt-4o-mini partially ignores it. A structural fix
   (passing only the user message, not the assistant response) would be more
   reliable but loses disambiguation context.

---

## 15. Final Recommendation

**Do NOT pass memory context to the micro-extractor.**

The proposal solves a real problem (the feedback loop) but at disproportionate
architectural cost. It violates import boundaries, couples the write path to the
read path, degrades extraction quality (per empirical evidence), and contradicts the
Single Responsibility Principle. Every production memory system studied (mem0,
Graphiti, cognee, NELL) separates extraction from reconciliation.

**Instead, strengthen the dedup pipeline:**

1. Lower same-type Jaccard threshold to 0.70 (catches the "User" vs "Carlos" case)
2. Add entity name normalization with configurable aliases
3. Replace the semantic similarity stub with real embedding cosine similarity

These changes are:
- Isolated to 2 files (`dedup.py`, `ingester.py`)
- Within existing architectural boundaries
- Independently testable
- Universally effective (catches duplicates from all sources)
- Supported by industry patterns

The micro-extractor should remain a stateless, best-effort, background extraction
component. Its job is to capture what was said. The ingester's job is to decide
what's worth keeping. These are separate concerns and should stay that way.

---

## References

### Production Frameworks
- mem0: arXiv:2504.19413, github.com/mem0ai/mem0
- Graphiti/Zep: arXiv:2501.13956, github.com/getzep/graphiti
- Letta/MemGPT: arXiv:2310.08560, github.com/letta-ai/letta
- cognee: github.com/topoteretes/cognee

### Empirical Research
- Context Rot (Chroma, 2025): trychroma.com/research/context-rot
- Lost in the Middle (Liu et al., 2024): arXiv:2307.03172
- Context Length Alone Hurts (EMNLP 2025): arXiv:2510.05381
- Better Prompts Hurt (2025): arXiv:2601.22025
- Incorrect Examples Harm (2024): arXiv:2404.12957

### Academic Systems
- NELL (CMU): cs.cmu.edu/~tom/pubs/NELL_aaai15.pdf
- Semantic Entity Resolution: blog.graphlet.ai/the-rise-of-semantic-entity-resolution

### Architectural Patterns
- CQRS: learn.microsoft.com/azure/architecture/patterns/cqrs
- Event Sourcing: martinfowler.com/eaaDev/EventSourcing.html
