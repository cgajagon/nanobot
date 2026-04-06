"""Memory store facade — coordinates subsystem modules.

``MemoryStore`` is a thin facade that composes focused subsystem modules:

- ``EventIngester`` — event write path (classify, dedup, merge, append)
- ``MemoryRetriever`` — retrieval read path (vector, BM25, reranking)
- ``ConsolidationPipeline`` — LLM-driven memory consolidation
- ``MemoryMaintenance`` — reindex, seed, health checks
- ``MemorySnapshot`` — rebuild and verify memory snapshots
Cross-cutting coordination (``get_memory_context``) stays on ``MemoryStore``.
Callers access subsystems directly for specific operations:
``store.ingester.append_events(...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.config.memory import MemoryConfig

from ._text import _to_str_list, _utc_now_iso, normalize_entity_name
from .consolidation_pipeline import ConsolidationPipeline
from .constants import PROFILE_KEYS
from .db import MemoryDatabase
from .db.alias_store import AliasRegistry
from .embedder import HashEmbedder, LocalEmbedder, OpenAIEmbedder
from .graph.entity_linker import ALIAS_MAP
from .graph.graph import KnowledgeGraph
from .maintenance import MemoryMaintenance
from .persistence.profile_io import ProfileStore
from .persistence.snapshot import MemorySnapshot
from .ranking.reranker import CompositeReranker, Reranker
from .read.context_assembler import ContextAssembler
from .read.graph_augmentation import GraphAugmenter
from .read.retrieval_planner import RetrievalPlanner
from .read.retriever import MemoryRetriever
from .read.scoring import RetrievalScorer
from .token_budget import DEFAULT_SECTION_WEIGHTS, TokenBudgetAllocator
from .write.classification import EventClassifier
from .write.coercion import EventCoercer
from .write.conflicts import ConflictManager
from .write.dedup import EventDeduplicator
from .write.extractor import MemoryExtractor
from .write.ingester import EventIngester

if TYPE_CHECKING:
    from nanobot.eval.memory_eval import EvalRunner
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session

    from .embedder import Embedder


class MemoryStore:
    """SQLite-backed memory store with structured profile/events maintenance."""

    def __init__(
        self,
        workspace: Path,
        *,
        memory_config: MemoryConfig | None = None,
        embedding_provider: str | None = None,
    ):
        self.workspace = workspace
        self._memory_config: MemoryConfig = memory_config or MemoryConfig()

        # Construct embedder: OpenAI → LocalEmbedder (ONNX) → HashEmbedder.
        # Explicit embedding_provider overrides the cascade.
        self._embedder: Embedder | None = None
        if embedding_provider in ("local", "onnx"):
            try:
                _local = LocalEmbedder()
                if _local.available:
                    self._embedder = _local
            except Exception:  # crash-barrier: local embedder init failure
                pass
        elif embedding_provider == "hash":
            self._embedder = HashEmbedder()
        else:
            # Default cascade: OpenAI → Local ONNX → Hash.
            try:
                _oai = OpenAIEmbedder()
                if _oai.available:
                    self._embedder = _oai
            except Exception:  # crash-barrier: OpenAI init failure
                pass
            if self._embedder is None:
                try:
                    _local = LocalEmbedder()
                    if _local.available:
                        self._embedder = _local
                except Exception:  # crash-barrier: local embedder init failure
                    pass
        if self._embedder is None:
            self._embedder = HashEmbedder()

        self.memory_dir: Path = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir: Path = self.memory_dir / "index"

        # Construct unified SQLite database.
        _dims = self._embedder.dims if self._embedder is not None else 384
        self.db: MemoryDatabase = MemoryDatabase(self.memory_dir / "memory.db", dims=_dims)

        self.retriever: MemoryRetriever  # set after graph/ingester init
        # EventIngester is constructed after graph is ready.
        # Deferred until graph is built below; placeholder here for type checkers.
        self.ingester: EventIngester  # set after graph init

        # Classifier and coercer are constructed early so the extractor can use them.
        self._classifier = EventClassifier()
        self._coercer = EventCoercer(self._classifier)

        self.extractor = MemoryExtractor(
            to_str_list=_to_str_list,
            coerce_event=self._coercer.coerce_event,
            utc_now_iso=_utc_now_iso,
        )
        # Profile manager — conflict_mgr and corrector wired after construction
        # (circular dependency: ConflictManager needs ProfileStore and vice versa).
        self.profile_mgr = ProfileStore(db=self.db)

        # Retrieval planner (LAN-207) — intent classification + policy + routing.
        self._planner = RetrievalPlanner()

        # Wire config section_weights into the budget allocator, falling back
        # to DEFAULT_SECTION_WEIGHTS when no overrides are provided.
        cfg_weights = self._memory_config.section_weights
        if cfg_weights:
            merged = dict(DEFAULT_SECTION_WEIGHTS)
            for intent, sw in cfg_weights.items():
                overrides = {k: v for k, v in sw.model_dump().items() if v > 0}
                base = dict(DEFAULT_SECTION_WEIGHTS.get(intent, {}))
                base.update(overrides)
                merged[intent] = base
            weights = merged
        else:
            weights = DEFAULT_SECTION_WEIGHTS
        self._budget_allocator = TokenBudgetAllocator(weights)

        # MemoryMaintenance: reindex, seed, health checks, backend stats.
        self.maintenance = MemoryMaintenance(
            db=self.db,
        )

        # Cross-encoder re-ranker.
        reranker_model = self._memory_config.reranker.model.strip()
        reranker_alpha = self._memory_config.reranker.alpha
        self._reranker: Reranker = CompositeReranker(alpha=reranker_alpha)
        if reranker_model.startswith("onnx:"):
            from .ranking.onnx_reranker import OnnxCrossEncoderReranker

            candidate = OnnxCrossEncoderReranker(
                model_name=reranker_model.split(":", 1)[1], alpha=reranker_alpha
            )
            if candidate.available:
                self._reranker = candidate
            else:
                logger.warning("ONNX reranker unavailable, using composite fallback")

        # Unified alias registry — single source of truth for entity aliases.
        self.alias_registry = AliasRegistry(self.db.alias_store)
        self._seed_alias_registry()
        self.alias_registry.load()

        # Knowledge graph (SQLite-backed via GraphStore).
        graph_enabled = self._memory_config.graph_enabled
        if graph_enabled:
            self.graph = KnowledgeGraph(db=self.db.graph_store, alias_registry=self.alias_registry)
        else:
            self.graph = KnowledgeGraph()  # disabled — all methods return empty

        # EventDeduplicator + EventIngester: own the full event write path.
        self._dedup = EventDeduplicator(
            coercer=self._coercer,
            conflict_pair_fn=self.profile_mgr._conflict_pair,
            alias_registry=self.alias_registry,
            embedder=self._embedder,
        )
        self.ingester = EventIngester(
            coercer=self._coercer,
            dedup=self._dedup,
            graph=self.graph,
            db=self.db.event_store,
        )

        # Conflict manager — now wire into ProfileStore (circular dep resolved
        # via post-construction wiring instead of lambda callbacks).
        self.conflict_mgr = ConflictManager(
            self.profile_mgr,
            memory_config=self._memory_config,
        )
        self.profile_mgr.set_conflict_mgr(self.conflict_mgr)

        # RetrievalScorer + GraphAugmenter + MemoryRetriever: read path.
        self._scorer = RetrievalScorer(
            profile_mgr=self.profile_mgr,
            reranker=self._reranker,
            memory_config=self._memory_config,
        )
        self._graph_aug = GraphAugmenter(
            graph=self.graph,
            extractor=self.extractor,
            read_events_fn=self.ingester.read_events,
        )
        self.retriever = MemoryRetriever(
            scorer=self._scorer,
            graph_aug=self._graph_aug,
            planner=self._planner,
            db=self.db.event_store,
            embedder=self._embedder,
        )

        # Context assembler (LAN-210) — prompt rendering extracted from MemoryStore.
        # Constructed after retriever/ingester/graph_aug so all dependencies
        # are direct references (no lambda callbacks needed).
        self._assembler = ContextAssembler(
            profile_mgr=self.profile_mgr,
            retriever=self.retriever,
            planner=self._planner,
            event_reader=self.ingester,
            graph_augmenter=self._graph_aug,
            budget_allocator=self._budget_allocator,
            db=self.db,
            embedder_available=self._embedder is not None,
        )

        # Evaluation / observability helper (LAN-204) — lazy-constructed on
        # first access so the eval package is not imported during normal agent
        # operation.  Only CLI memory commands access this property.
        self._eval_runner: EvalRunner | None = None

        # MemorySnapshot: rebuild memory snapshots + verify integrity.
        self.snapshot = MemorySnapshot(
            profile_mgr=self.profile_mgr,
            profile_section_lines_fn=self._assembler._profile_section_lines,
            recent_unresolved_fn=self._assembler._recent_unresolved,
            profile_keys=PROFILE_KEYS,
            db=self.db,
        )

        # CorrectionOrchestrator — wire into ProfileStore (post-construction).
        from .persistence.profile_correction import (
            CorrectionOrchestrator as _CorrectionOrchestrator,
        )

        self._corrector = _CorrectionOrchestrator(
            profile_store=self.profile_mgr,
            extractor=self.extractor,
            ingester=self.ingester,
            coercer=self._coercer,
            conflict_mgr=self.conflict_mgr,
            snapshot=self.snapshot,
            embedder=self._embedder,
        )
        self.profile_mgr.set_corrector(self._corrector)

        # ConsolidationPipeline: full consolidate workflow (LAN-215).
        self._consolidation = ConsolidationPipeline(
            extractor=self.extractor,
            ingester=self.ingester,
            profile_mgr=self.profile_mgr,
            conflict_mgr=self.conflict_mgr,
            snapshot=self.snapshot,
            db=self.db,
            embedder=self._embedder,
        )

    # ------------------------------------------------------------------
    # Alias registry seeding
    # ------------------------------------------------------------------

    def _seed_alias_registry(self) -> None:
        """Seed the alias registry from config, entity_linker, and graph aliases."""
        store = self.db.alias_store

        # Source 1: config user_aliases (highest confidence)
        for alias in self._memory_config.user_aliases:
            store.register(normalize_entity_name(alias), "_user_", confidence=1.0, source="config")

        # Source 2: static entity_linker map
        for alias, canonical in ALIAS_MAP.items():
            store.register(
                normalize_entity_name(alias),
                normalize_entity_name(canonical),
                confidence=0.9,
                source="linker",
            )

        # Source 3: existing graph entity aliases (if graph enabled)
        if self._memory_config.graph_enabled:
            rows = self.db.graph_store.search_entities("", limit=1000)
            if len(rows) >= 1000:
                logger.warning(
                    "Graph entity seed truncated at 1000; some aliases may not be registered"
                )
            for row in rows:
                canonical = str(row.get("name", ""))
                aliases_text = str(row.get("aliases", ""))
                if aliases_text:
                    for alias in aliases_text.split(","):
                        alias = alias.strip()
                        if alias:
                            store.register(
                                normalize_entity_name(alias),
                                normalize_entity_name(canonical),
                                confidence=0.8,
                                source="graph",
                            )

    # ------------------------------------------------------------------
    # Computed properties and internal callbacks
    # ------------------------------------------------------------------

    @property
    def memory_config(self) -> MemoryConfig:
        """Typed memory configuration."""
        return self._memory_config

    @property
    def eval_runner(self) -> EvalRunner:
        """Lazy-constructed EvalRunner — only imported when accessed (CLI commands)."""
        if self._eval_runner is None:
            from nanobot.eval.memory_eval import EvalRunner

            self._eval_runner = EvalRunner(
                retriever=self.retriever,
                workspace=self.workspace,
                memory_dir=self.memory_dir,
                memory_config=self._memory_config,
                maintenance=self.maintenance,
            )
        return self._eval_runner

    # ------------------------------------------------------------------
    # get_memory_context — facade method
    # ------------------------------------------------------------------

    async def get_memory_context(
        self,
        *,
        query: str | None = None,
        retrieval_k: int = 6,
        token_budget: int = 900,
        memory_md_token_cap: int = 1500,
        mode: str | None = None,
    ) -> str:
        return await self._assembler.build(
            query=query,
            retrieval_k=retrieval_k,
            token_budget=token_budget,
            memory_md_token_cap=memory_md_token_cap,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Consolidation pipeline — delegated to ConsolidationPipeline
    # ------------------------------------------------------------------

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
        memory_mode: str | None = None,
        enable_contradiction_check: bool = True,
    ) -> bool:
        """Consolidate old messages into SQLite snapshots and history via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        return await self._consolidation.consolidate(
            session,
            provider,
            model,
            archive_all=archive_all,
            memory_window=memory_window,
            memory_mode=memory_mode,
            enable_contradiction_check=enable_contradiction_check,
        )
