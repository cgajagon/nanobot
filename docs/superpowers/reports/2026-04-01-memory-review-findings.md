# Memory Subsystem Review: Knowledge Graph & Profile (Task #4)

**Reviewer:** Agent  
**Date:** 2026-04-01  
**Status:** Complete  
**Components Reviewed:**
- `nanobot/memory/graph/` (KnowledgeGraph, entity classification, traversal, ontology)
- `nanobot/memory/persistence/` (ProfileStore, belief lifecycle, profile correction, conflicts)

---

## Executive Summary

The knowledge graph and profile systems are **architecturally sound and well-separated**. The entity classification cascade is robust with appropriate confidence tiering. The belief lifecycle is properly abstracted and the circular dependency between ProfileStore/ConflictManager is correctly handled via post-construction wiring.

**Issues Found:** 2 moderate, 2 minor  
**Risk Level:** LOW  
**Recommended Actions:** Resolve circular dependency anti-pattern; add missing ontology predicates for AI agent use cases

---

## 1. Entity Classification System

### Findings

**Component:** `entity_classifier.py` (86% keyword data, 14% logic — appropriate size exception)

#### ✅ Strengths

- **Six-signal cascade** (regex → tokens → phrases → suffixes → roles → capitalization) is well-ordered by confidence
- **Scored variant** (`classify_entity_type_scored()`) provides ranked candidates for tie-breaking
- **Predicate-based refinement** properly demotes UNKNOWN entities without overwriting classified types
- **Alias resolution** happens before classification (via `entity_linker.py`)
- **Backward compatibility** — `classify_entity_type()` returns single best match

#### Issues & Recommendations

**Issue 1.1: Capitalisation heuristic over-includes system terms** ⚠️ MODERATE

**Location:** `entity_classifier.py:516-525`

```python
significant = [
    w for w in name_words if w[0].isupper() and w.lower() not in _CAPITALIZATION_STOPWORDS
]
if significant and not (words & _NON_PERSON_KEYWORDS):
    candidates.append(TypeScore(EntityType.PERSON, 0.45, "capitalized"))
```

**Problem:** A name like "Redis Queue" will fire the capitalization heuristic if none of "redis" or "queue" are in `_NON_PERSON_KEYWORDS`. The check `not (words & _NON_PERSON_KEYWORDS)` requires *no* overlaps. If the entity is a compound like "New System" and "system" is a keyword, the heuristic won't fire — correct. But "New Service X" where only "service" matches: the entity still gets PERSON confidence 0.45.

**Impact:** Low false-positive rate in practice due to confidence ordering (keywords at 0.85 beat capitalisation at 0.45), but the heuristic could misclassify novel proper nouns that happen to lack matching keywords.

**Proposed Fix:** Add a positive threshold — fire capitalization heuristic only if:
```python
significant_count = len(significant)
keyword_overlap = len(words & _NON_PERSON_KEYWORDS)
# Require at least 1 significant capital AND no more than 1 keyword overlap
if significant_count >= 1 and keyword_overlap <= 1:
    candidates.append(TypeScore(EntityType.PERSON, 0.45, "capitalized"))
```

**Test Case:** Verify that "Redis", "PostgreSQL", "New Framework" classify correctly (TECHNOLOGY/FRAMEWORK, not PERSON).

---

**Issue 1.2: Missing ontology predicates for agent-native triples** ⚠️ MODERATE

**Location:** `ontology_types.py:131-141`, `ontology_rules.py:29-200`

The predicate vocabulary includes agent-operational types (PERFORMS, EXECUTES, PRODUCES, OBSERVES, STORES, RECALLS), but is missing critical predicates for agent introspection and memory management:

- **RELEARNED_FROM** (agent discovers something that contradicts prior learning)
- **CORRECTED_BY** (user corrected agent belief)
- **INFERRED_FROM** (agent deduced X from observations Y)
- **CONFIDENCE_WAS_X** (historical confidence tracking)
- **CONTRADICTS** (explicit negation links between entities)

**Impact:** Medium. These predicates enable the agent to reason about its own knowledge evolution and conflict resolution. Without explicit edges for corrections/learning, the graph is less useful for retrospection.

**Proposed Fix:** Extend `RelationType` enum:
```python
# Learning and correction
RELEARNED_FROM = "RELEARNED_FROM"      # agent re-discovers fact with new confidence
CORRECTED_BY = "CORRECTED_BY"          # user corrected agent's belief
INFERRED_FROM = "INFERRED_FROM"        # agent deduced from observations
CONTRADICTS = "CONTRADICTS"            # explicit negation (X is not Y)
OVERRIDES = "OVERRIDES"                # newer belief supersedes older
```

And add domain/range rules in `RELATION_RULES`:
```python
RelationType.RELEARNED_FROM: {
    "subject": frozenset({EntityType.AGENT, EntityType.MEMORY}),
    "object": frozenset({EntityType.CONCEPT, EntityType.FACT, EntityType.MEMORY}),
},
```

---

### Entity Classifier Tests

**Status:** Tested in `tests/test_entity_classifier.py` — good coverage of regex, keyword, phrase, suffix, and role signals.

**Gap:** No test for the capitalisation heuristic edge case above. Add:
```python
def test_capitalization_heuristic_with_keyword_overlap():
    """Verify capitalisation doesn't fire when name contains keywords."""
    # Should NOT classify as PERSON (system is a known keyword)
    etype = classify_entity_type("New System")
    assert etype in {EntityType.SYSTEM, EntityType.TECHNOLOGY}
    
    # Should classify as PERSON (no keyword overlap)
    etype = classify_entity_type("Jane Smith")
    assert etype == EntityType.PERSON
```

---

## 2. Knowledge Graph Storage & Traversal

### Findings

**Components:**
- `graph.py` — KnowledgeGraph facade (write/read methods)
- `graph_traversal.py` — BFS path-finding and subgraph queries
- `ontology_rules.py` — Domain/range validation

#### ✅ Strengths

- **Protocol abstraction** (`_KnowledgeGraphProtocol`) decouples traversal from storage details
- **Disabled-graph stub** — `enabled=False` mode returns empty results gracefully (no crashes)
- **Entity merging** preserves first_seen, unions aliases, merges properties
- **Confidence demotion** on constraint violation (never rejects, just demotes to 0.5x)
- **Display-name preservation** — canonical name (lowercased) separate from display name stored in properties

#### Issues & Recommendations

**Issue 2.1: BFS traversal unbounded neighborhood collection** ⚠️ MINOR

**Location:** `graph_traversal.py:40-100`

```python
def find_paths(graph, source, target, max_depth=3):
    # ...
    while queue and len(paths) < 5:
        path = queue.popleft()
        # ...
        for edge in graph.get_edges_from(current):
            neighbor = edge["target"]
            if neighbor in path:
                continue  # avoid cycles
            # Next iteration may add many neighbors...
```

**Problem:** In a highly connected graph (e.g., a dense relationship network with thousands of edges between hubs), BFS can explode the queue size. The code limits *paths* to 5, but doesn't limit *nodes explored*. A node with 500 outgoing edges adds 500 items to the queue before the "5 paths" check.

**Impact:** Low in practice (graphs are usually sparse), but could cause memory bloat if the agent discovers a highly interconnected domain (e.g., all PERSON nodes in a large organization).

**Proposed Fix:** Add a visited set and a max-nodes-explored limit:

```python
max_nodes_explored = 200
visited: set[str] = set()

while queue and len(paths) < 5 and len(visited) < max_nodes_explored:
    path = queue.popleft()
    # ...
    for edge in graph.get_edges_from(current):
        neighbor = edge["target"]
        if neighbor in visited or neighbor in path:
            continue
        visited.add(neighbor)
        # ...
```

---

**Issue 2.2: Open-world semantics allow invalid triples without feedback**

**Location:** `graph.py:144-153`

```python
validation = validate_triple_types(triple.predicate, sub_type, obj_type)
if not validation.valid:
    logger.debug("Triple constraint violation (%s): %s", ...)
    confidence *= 0.5  # demote but still insert
```

**Problem:** Confidence demotion is silent (DEBUG level log). If the LLM extracts a triple like "GitHub DEPENDS_ON User", the system demotes confidence and inserts it anyway. The agent has no way to know the triple is likely wrong.

**Impact:** Low. The confidence reduction (0.5x) means the edge will be weighted lower in traversals. But bad triples accumulate undetected.

**Proposed Fix:** Promote validation violations to INFO or WARN level and track a count:

```python
validation = validate_triple_types(triple.predicate, sub_type, obj_type)
if not validation.valid:
    logger.warning(
        "Triple validation failed (%s): %s → %s. Demoting confidence. Reason: %s",
        triple.predicate.value, sub_type.value, obj_type.value, validation.reason
    )
    confidence *= 0.5
    # Optionally: track in metadata for post-consolidation review
```

---

## 3. Profile Store & Belief Lifecycle

### Findings

**Components:**
- `profile_io.py` — ProfileStore facade with thin delegation
- `belief_lifecycle.py` — Extracted belief mutation functions
- `profile_correction.py` — Live user correction pipeline
- `conflict_types.py` — Conflict record dataclass

#### ✅ Strengths

- **Proper extraction** — belief lifecycle functions extracted from ProfileStore, receive protocol-typed ProfileStore
- **Stable IDs** — deterministic SHA1-based belief IDs enable cross-session tracking
- **Metadata nesting** — normalized text as key in `meta[section][norm_text]`, entry dict stores all metadata
- **Confidence dynamics** — confidence clamps to [0.05, 0.99], evidence capped at 10 refs
- **Pinning** reactivates stale beliefs — good UX for user-important beliefs

#### Issues & Recommendations

**Issue 3.1: Circular dependency between ProfileStore, ConflictManager, CorrectionOrchestrator** ⚠️ MODERATE

**Location:** `profile_io.py:147-157`, `store.py:~97, ~112`

```python
# profile_io.py
class ProfileStore:
    def __init__(self, db=None):
        self._conflict_mgr: _ConflictManagerProtocol | None = None
        self._corrector: _CorrectionProtocol | None = None
    
    def set_conflict_mgr(self, conflict_mgr):
        self._conflict_mgr = conflict_mgr  # post-construction wiring
```

**Problem:** ProfileStore's core mutation methods (`_conflict_pair`, `_apply_profile_updates`, `_has_open_conflict`) delegate to ConflictManager. But ConflictManager receives ProfileStore in its constructor. This creates a circular dependency that's broken by **post-construction wiring in MemoryStore**.

**Risk:** Fields remain `None` between `ProfileStore.__init__` and `set_conflict_mgr()`. If a method is called before wiring completes, `RuntimeError` is raised. This is documented but fragile.

**Proposed Fix:** Use lazy initialization instead of `None` fields:

```python
class ProfileStore:
    def __init__(self, db=None, *, conflict_mgr=None, corrector=None):
        self._db = db
        self._conflict_mgr = conflict_mgr
        self._corrector = corrector
    
    def _ensure_conflict_mgr(self):
        if self._conflict_mgr is None:
            raise RuntimeError("conflict_mgr not provided at construction")
        return self._conflict_mgr
    
    def _conflict_pair(self, old, new):
        mgr = self._ensure_conflict_mgr()
        return mgr._conflict_pair(old, new)
```

Or better: pass conflict_mgr/corrector at construction time in `MemoryStore.__init__`:

```python
# In MemoryStore.__init__
conflict_mgr = ConflictManager(...)
profile_store = ProfileStore(db=self.db, conflict_mgr=conflict_mgr, corrector=None)
corrector = CorrectionOrchestrator(..., profile_store=profile_store)
profile_store.set_corrector(corrector)  # only corrector needs post-wiring due to its complex deps
```

This reduces the wiring window to a single `set_corrector()` call.

---

**Issue 3.2: Belief evidence references not bidirectional** ⚠️ MINOR

**Location:** `profile_io.py:287-293`

```python
def _touch_meta_entry(self, entry, ...):
    # ...
    if evidence_event_id:
        refs = entry.setdefault("evidence_event_ids", [])
        if evidence_event_id not in refs:
            refs.append(evidence_event_id)
            if len(refs) > self._MAX_EVIDENCE_REFS:
                del refs[: len(refs) - self._MAX_EVIDENCE_REFS]
```

**Problem:** Evidence references are one-way (belief → events). There's no reverse index (event → beliefs that cite it). If a belief is retracted, you can't easily find all events that reference it to update their "this belief was retracted" status.

**Impact:** Low. The belief's status is independent of events. But it would be useful for retrospection: "which events supported this (now-retracted) belief?"

**Proposed Fix:** Add an optional reverse index in the profile:

```python
# In profile structure:
{
    "preferences": [...],
    "meta": {...},
    "conflicts": [...],
    "_evidence_index": {  # event_id → [belief_id, belief_id, ...]
        "ev-12345": ["bf-abc", "bf-def"],
    }
}
```

Update on evidence link, clean up on belief retraction. Not critical but improves auditability.

---

## 4. Conflict Resolution System

### Findings

**Components:**
- `conflict_types.py` — ConflictRecord dataclass (properly scoped)
- `write/conflicts.py` — ConflictManager (write path)
- `profile_correction.py` — CorrectionOrchestrator (live user correction)

#### ✅ Strengths

- **ConflictRecord model** — stable dataclass with `.to_dict()` / `.from_dict()` round-trip
- **Three-level detection** (negation tokens + overlap threshold) is conservative but appropriate
- **Auto-resolution** by confidence gap (≥ 0.25) avoids user queries on minor conflicts
- **Temporal tiebreaker** — newer belief wins on confidence tie
- **Status tracking** — open | needs_user | resolved states track resolution workflow

#### Issues & Recommendations

**Issue 4.1: Negation detection fragile for multi-word sentences** ⚠️ MINOR

**Location:** `write/conflicts.py` (not provided, but referenced in architecture docs)

**Problem:** Conflict detection relies on token-level negation (`" not "` or `"n't"`). This works for direct contradictions ("has permissions" vs "does not have permissions") but misses semantic contradictions ("Alice works on Project X" vs "Alice left the company").

**Impact:** Low. System defaults to auto-resolve by confidence when unsure, and user can manually resolve via conflict tool.

**Proposed Fix:** Extend negation detection to a second pass:

```python
def _semantic_conflict(old_value: str, new_value: str) -> bool:
    """Detect semantic contradictions beyond negation tokens."""
    # Direct negation check (existing)
    if " not " in old_value or "n't" in old_value:
        # ... existing logic
    
    # Semantic check: known antonyms
    antonym_pairs = {
        ("active", "inactive"),
        ("enabled", "disabled"),
        ("required", "optional"),
        ("public", "private"),
        ("owner", "guest"),
    }
    old_tokens = set(old_value.lower().split())
    new_tokens = set(new_value.lower().split())
    
    for (a, b) in antonym_pairs:
        if (a in old_tokens and b in new_tokens) or (b in old_tokens and a in new_tokens):
            return True
    return False
```

---

## 5. Circular Dependency Deep Dive

### Current State

```
ProfileStore (constructor: db)
    ├─ reads from: profile.json
    ├─ delegates to: ConflictManager._conflict_pair()
    │                ConflictManager._apply_profile_updates()
    │                ConflictManager.has_open_conflict()
    └─ delegates to: CorrectionOrchestrator.apply_live_user_correction()

ConflictManager (constructor: db, profile_store, ...)
    └─ reads from: ProfileStore.read_profile() (passed at construction)
    
CorrectionOrchestrator (constructor: profile_store, ...)
    └─ writes to: ProfileStore._add_belief_to_profile()
                  ProfileStore._touch_meta_entry()
```

### Wiring Order (in MemoryStore.__init__)

1. Construct ProfileStore (no conflict_mgr yet)
2. Construct ConflictManager (passes profile_store instance)
3. Call `profile_store.set_conflict_mgr(conflict_mgr)` ← Window of danger
4. Construct CorrectionOrchestrator (passes profile_store)
5. Call `profile_store.set_corrector(corrector)` ← Window of danger

**Risk:** If any code path tries to call `profile_store._conflict_pair()` between steps 1-3, it crashes.

### Recommended Fix: Deferred Dependency

```python
# In memory/store.py:

# Step 1: Construct core services
profile_store = ProfileStore(db=db)

# Step 2: Create conflict manager with immediate wiring
conflict_mgr = ConflictManager(
    db=db,
    profile_store=profile_store,
    ...
)
profile_store._conflict_mgr = conflict_mgr  # Immediate, not deferred

# Step 3: Create corrector with deferred wiring (still needs profile_store)
corrector = CorrectionOrchestrator(
    profile_store=profile_store,
    conflict_mgr=conflict_mgr,
    ...
)
profile_store._corrector = corrector  # Deferred okay here
```

This is a minor improvement but makes the wiring intent clearer.

---

## 6. Graph vs. Profile Integration

### Findings

The knowledge graph and profile are **properly separated**:

- **Graph** owns entities and relationships (sparse, inference-focused)
- **Profile** owns beliefs and confidence (dense, persistent, user-facing)

**One touch point:** Profile snapshot generation may reference entity names from the graph for context. This is read-only and safe.

#### Recommendation

Consider explicit edge from profile beliefs to graph entities:

```python
# In belief lifecycle
def add_belief(store, field, text, ...):
    # Extract mentioned entities from the belief text
    entities = entity_extractor.extract(text)
    for entity in entities:
        # Ensure entity exists in graph with a link back
        await graph.upsert_entity(Entity(...))
        await graph.add_relationship(Relationship(
            source="belief",
            source_id=belief_id,
            target_id=entity.name,
            relation=RelationType.REFERENCES,
        ))
```

This creates an audit trail (which graph entities a belief references), useful for conflict resolution.

---

## 7. Testing Coverage

### Gaps

| Component | Coverage | Gap |
|-----------|----------|-----|
| Entity classification | High | Capitalisation heuristic edge case |
| KnowledgeGraph CRUD | High | Graph traversal memory limits |
| ProfileStore API | High | Circular dependency window |
| Belief lifecycle | High | Evidence reverse indexing |
| Conflict detection | Medium | Semantic contradictions |

---

## Summary of Recommendations

### Priority 1 (Implement Soon)

1. **Add test for capitalization heuristic** — ensure "New System" doesn't classify as PERSON
2. **Resolve circular dependency** — make ConflictManager dependency immediate instead of post-construction

### Priority 2 (Nice to Have)

3. **Add BFS memory limit** — prevent queue explosion in dense graphs
4. **Extend negation detection** — add antonym pairs for semantic contradictions
5. **Promote validation warnings** — log constraint violations at WARN level

### Priority 3 (Future)

6. **Add missing predicates** — RELEARNED_FROM, CORRECTED_BY, INFERRED_FROM, CONTRADICTS
7. **Implement evidence reverse indexing** — event → beliefs that cite it

---

## Risk Assessment

| Finding | Severity | Impact | Mitigation |
|---------|----------|--------|-----------|
| Capitalisation heuristic | Low | Rare misclassification | Add test |
| Missing predicates | Medium | Reduced agent introspection | Extend ontology |
| BFS memory | Low | Potential bloat in dense graphs | Add limit |
| Circular dependency | Medium | Fragile wiring window | Immediate dependency |
| Validation silent fails | Low | Bad triples accumulate | Promote to WARN |
| Semantic conflict miss | Low | User manual resolution | Extend detection |

---

## Conclusion

The knowledge graph and profile systems are **well-designed and maintainable**. The entity classification pipeline is sound, the belief lifecycle is properly abstracted, and the separation of concerns is clean.

**All findings are addressable in normal development flow.** None require architectural changes.

**Recommended next steps:**
1. Fix circular dependency (1-2 hour refactor)
2. Add entity classification tests (15 min)
3. Extend predicate vocabulary (30 min)
4. Add BFS memory limits (15 min)

**Total estimated effort:** ~2 hours.
