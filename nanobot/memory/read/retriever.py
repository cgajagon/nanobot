"""Memory retrieval read path — unified vector + FTS5 + RRF pipeline.

``MemoryRetriever`` orchestrates the retrieval pipeline by delegating to
``RetrievalScorer`` (filter/score/rerank) and ``GraphAugmenter``
(entity collection and graph context).

Pipeline architecture (pipes and filters)::

    Source (vector + FTS5 + RRF) → Graph Augment → Filter → Score → Rerank → Truncate
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from nanobot.observability.langfuse import retriever_span
from nanobot.observability.tracing import bind_trace

from .._text import _to_datetime
from .graph_augmentation import GraphAugmenter
from .retrieval_planner import RetrievalPlanner
from .retrieval_types import RetrievedMemory, retrieved_memory_from_dict
from .scoring import RetrievalScorer

if TYPE_CHECKING:
    from ..db.event_store import EventStore
    from ..embedder import Embedder


def filter_expired(items: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Exclude events whose TTL has expired.

    Uses ``last_confirmed`` (if available) or ``timestamp`` for age calculation.
    Events without ``ttl_days`` or with invalid TTL always pass through.
    """
    filtered: list[dict[str, Any]] = []
    for item in items:
        ttl = item.get("ttl_days")
        if not isinstance(ttl, int) or ttl <= 0:
            filtered.append(item)
            continue
        ts_str = str(item.get("last_confirmed") or item.get("timestamp", ""))
        ts = _to_datetime(ts_str)
        if ts is None:
            filtered.append(item)
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=now.tzinfo)
        age_days = (now - ts).total_seconds() / 86400.0
        if age_days <= ttl:
            filtered.append(item)
    return filtered


class MemoryRetriever:
    """Orchestrates the full memory retrieval read path.

    Delegates scoring to ``RetrievalScorer`` and graph augmentation to
    ``GraphAugmenter``.  Owns only fusion, metadata enrichment, and
    top-level orchestration.
    """

    def __init__(
        self,
        *,
        scorer: RetrievalScorer,
        graph_aug: GraphAugmenter,
        planner: RetrievalPlanner,
        db: EventStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._scorer = scorer
        self._graph_aug = graph_aug
        self._planner = planner
        self._db = db
        self._embedder = embedder

    def _vector_weight(self) -> float:
        """RRF vector weight adapted to embedder semantic quality.

        Returns a lower weight when the embedder produces lower-quality
        vectors, so FTS5 (keyword matching) dominates the fusion score.
        """
        if self._embedder is None:
            return 0.0
        return self._embedder.vector_quality

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
    ) -> list[RetrievedMemory]:
        self._graph_aug.reset_cache()
        t0 = time.monotonic()

        async with retriever_span(
            name="memory_retrieve",
            input={"query": query, "top_k": top_k},
        ) as obs:
            # Unified path: vector + FTS5 + RRF when db and embedder are injected
            if self._db is not None and self._embedder is not None:
                results = await self._retrieve_unified(
                    query,
                    top_k=top_k,
                    t0=t0,
                )
            else:
                results = []

            if obs is not None:
                obs.update(
                    output=f"{len(results)} results",
                    metadata={
                        "result_count": len(results),
                        "duration_ms": round((time.monotonic() - t0) * 1000),
                    },
                )

            return results

    # ------------------------------------------------------------------
    # Unified path (vector + FTS5 + RRF)
    # ------------------------------------------------------------------

    async def _retrieve_unified(
        self,
        query: str,
        *,
        top_k: int,
        t0: float,
    ) -> list[RetrievedMemory]:
        """Single fused retrieval: vector + FTS5 + RRF.

        Used when ``EventStore`` and ``Embedder`` are injected.  Runs
        embedding and dual-source search (vector KNN + FTS5), fuses via
        Reciprocal Rank Fusion, then applies the standard scoring pipeline.
        """
        if self._db is None:
            raise RuntimeError("EventStore not initialized — db is None")
        if self._embedder is None:
            raise RuntimeError("Embedder not initialized — embedder is None")

        embedder = self._embedder
        db = self._db

        plan = self._planner.plan(query)
        policy = plan.policy
        candidate_k = max(1, min(top_k * int(policy.get("candidate_multiplier", 3)), 60))

        # 1. Embed + FTS concurrently (FTS does not need the vector)
        async def _safe_embed() -> list[float] | None:
            try:
                return await embedder.embed(query)
            except Exception:  # crash-barrier: degrade to FTS-only on embed failure
                bind_trace().warning("Embedding failed, falling back to FTS-only retrieval")
                return None

        query_vec, fts_results = await asyncio.gather(
            _safe_embed(),
            asyncio.to_thread(db.search_fts, query, candidate_k),
        )

        # 2. Vector search only if embedding succeeded
        if query_vec is not None:
            vec_results = await asyncio.to_thread(db.search_vector, query_vec, candidate_k)
        else:
            vec_results = []

        # 3. Fuse via RRF
        candidates = self._fuse_results(
            vec_results, fts_results, vector_weight=self._vector_weight()
        )

        if not candidates:
            candidates = await asyncio.to_thread(db.read_events, limit=candidate_k)
            if not candidates:
                bind_trace().debug(
                    "Memory retrieve source=unified results=0 duration_ms={:.0f}",
                    (time.monotonic() - t0) * 1000,
                )
                return []

        # 4. Enrich metadata (still dict-based for scorer compatibility)
        self._enrich_item_metadata(candidates)

        # 5. Filter
        filtered, _filter_counts = self._scorer.filter_items(candidates, plan)

        # 5b. TTL expiry filter
        filtered = filter_expired(filtered, datetime.now(timezone.utc))

        # 6. Score
        profile_data = self._scorer.load_profile_scoring_data()
        graph_entities = self._graph_aug.collect_graph_entity_names(
            query, self._graph_aug.read_events()
        )
        scored = self._scorer.score_items(
            filtered,
            plan,
            profile_data,
            graph_entities,
            use_recency=True,
            router_enabled=True,
            type_separation_enabled=True,
        )

        # 7. Rerank
        scored = self._scorer.rerank_items(query, scored)

        # 8. Sort + truncate
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        scored = scored[:top_k]

        # 9. Convert to typed objects
        results = [retrieved_memory_from_dict(item) for item in scored]

        bind_trace().debug(
            "Memory retrieve source=unified results={} duration_ms={:.0f}",
            len(results),
            (time.monotonic() - t0) * 1000,
        )
        return results

    @staticmethod
    def _fuse_results(
        vec_results: list[dict[str, Any]],
        fts_results: list[dict[str, Any]],
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of vector and FTS5 results."""
        k = 60  # standard RRF constant
        scores: dict[str, float] = {}
        items: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(vec_results):
            eid = str(item.get("id", ""))
            scores[eid] = scores.get(eid, 0.0) + vector_weight / (k + rank)
            items[eid] = item

        for rank, item in enumerate(fts_results):
            eid = str(item.get("id", ""))
            scores[eid] = scores.get(eid, 0.0) + (1 - vector_weight) / (k + rank)
            if eid not in items:
                items[eid] = item

        # Sort by fused score descending
        ranked = sorted(scores.keys(), key=lambda eid: scores[eid], reverse=True)
        result: list[dict[str, Any]] = []
        for eid in ranked:
            entry = dict(items[eid])
            entry["_rrf_score"] = scores[eid]
            entry["score"] = scores[eid]
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Pipeline stage: metadata enrichment
    # ------------------------------------------------------------------

    def _enrich_item_metadata(self, items: list[dict[str, Any]]) -> None:
        """Promote metadata fields (topic, stability, memory_type) to top level."""
        import json as _json

        for item in items:
            memory_type = RetrievalPlanner.memory_type_for_item(item)
            item["memory_type"] = memory_type
            meta = item.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = _json.loads(meta)
                except (ValueError, TypeError):
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}
            # Unpack extra fields stored by ingester (entities, triples, etc.)
            extras = meta.pop("_extra", None)
            if isinstance(extras, dict):
                for k, v in extras.items():
                    if k not in item:
                        item[k] = v
            if not item.get("topic"):
                item["topic"] = str(meta.get("topic", "")).strip()
            if not item.get("stability"):
                item["stability"] = str(meta.get("stability", "medium")).strip()
