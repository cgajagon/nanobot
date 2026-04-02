# Strategy Feedback Loop Completion — Design Spec

**Goal:** Close the procedural memory feedback loop so strategies strengthen on success, weaken on failure, and display with directive formatting that LLMs follow.

**Problem:** Strategies are extracted and saved correctly but (1) displayed with weak language the LLM ignores, (2) confidence never updates post-turn, and (3) no tracking of which strategies were in the prompt.

## Changes

### 1. Directive prompt formatting (`context/context.py:128-138`)

Replace the current weak formatting:
```
# Relevant Strategies
These strategies were learned from past sessions. Apply them when relevant.

**obsidian / empty_recovery:obsidian_search** (confidence: 50%, used 0x)
WHEN: looking up a project code...
```

With directive formatting:
```
# Tool-Use Rules (from past sessions)

These rules correct known failure patterns. Follow them before choosing tools.

⚠️ WHEN: looking up a project code in Obsidian
  DON'T: obsidian search (only matches file content, not folder/file names)
  DO: obsidian vault → browse by folder name
  [confidence: 80%, applied 5x]
```

Changes:
- Title: "Tool-Use Rules" not "Relevant Strategies"
- Instruction: "Follow them before choosing tools" not "Apply when relevant"
- Warning marker `⚠️` on each rule
- Confidence de-emphasized in brackets at end
- Domain/task_type metadata removed from display (adds noise)

### 2. Track loaded strategies (`context/context.py`)

Add `self._last_loaded_strategies: list[Strategy] = []` to ContextBuilder. After retrieving strategies in `build_system_prompt()`, store them. Expose via property `last_loaded_strategies`.

This allows the message processor to know which strategies were in the prompt for confidence updates.

### 3. Wire confidence updates (`agent/message_processor.py:370+`)

After the existing strategy extraction block (line 370), add confidence update logic:

```python
# Update confidence for strategies that were in context
if self._strategy_extractor and self._context:
    loaded = self._context.last_loaded_strategies
    if loaded:
        had_guardrails = bool(_guardrail_acts)
        self._strategy_extractor.update_confidence(
            loaded, had_guardrail_activations=had_guardrails,
        )
```

This runs after every turn (not just turns with guardrails), because successful turns should strengthen loaded strategies.

### 4. Prune low-confidence strategies

`StrategyAccess.prune(min_confidence=0.1)` is already called in `agent_factory.py` at startup. No change needed — strategies that decay below 0.1 get cleaned up on next session start.

## What we're NOT changing

- `StrategyExtractor._llm_summarize()` — already produces WHEN/DON'T/DO format
- `StrategyExtractor._build_strategy()` — extraction logic is correct
- `StrategyAccess.retrieve()` filtering — no domain filtering (YAGNI)
- `StrategyAccess` CRUD methods — all work correctly
- Strategy table schema — no changes needed

## Files to modify

| File | Change |
|------|--------|
| `nanobot/context/context.py:128-138` | Directive formatting + track loaded strategies |
| `nanobot/agent/message_processor.py:370+` | Wire confidence update call |
| `tests/test_context_builder.py` or equivalent | Test new formatting + strategy tracking |
| `tests/test_message_processor.py` or equivalent | Test confidence wiring |

## Testing

- Unit test: ContextBuilder formats strategies with directive markers
- Unit test: ContextBuilder.last_loaded_strategies returns what was loaded
- Unit test: MessageProcessor calls update_confidence post-turn
- Contract test: confidence increases after successful turn with strategies
- Contract test: confidence decreases after guardrail-activated turn with strategies
