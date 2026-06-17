"""
Hybrid retrieval combining TF-IDF, BM25, and dense embeddings.

Two retrieval modes
-------------------
parallel (fusion)
    All three models run independently on the full corpus and their
    ranked lists are merged using either Reciprocal Rank Fusion (RRF)
    or a weighted linear combination.

serial (pipeline)
    BM25 produces a coarse candidate set (top-100); the embedding model
    re-ranks only those candidates.  TF-IDF is not used in serial mode.

Fusion methods
--------------
rrf     Reciprocal Rank Fusion — rank-based, no score normalisation needed.
linear  Min-max normalise each list then sum weighted scores.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

from .bm25_model import BM25Model
from .embedding_model import EmbeddingModel
from .tfidf_model import TFIDFModel

logger = logging.getLogger(__name__)

_CANDIDATE_K = 100  # coarse retrieval pool size


class HybridModel:
    """Hybrid retrieval model combining sparse and dense signals.

    Args:
        mode: Default retrieval mode — ``"parallel"`` (fusion) or
            ``"serial"`` (BM25 → embedding re-rank).
        fusion_method: Score merging strategy when ``mode="parallel"`` —
            ``"rrf"`` (Reciprocal Rank Fusion) or ``"linear"`` (weighted sum).
        bm25_weight: Weight applied to BM25 scores in linear fusion.
        embedding_weight: Weight applied to embedding scores in linear fusion.
        tfidf_weight: Weight applied to TF-IDF scores in linear fusion.
        rrf_k: RRF smoothing constant (typical values 10–100; default 60).
        embedding_model_name: Model identifier forwarded to
            :class:`EmbeddingModel`.
    """

    def __init__(
        self,
        mode: str = "parallel",
        fusion_method: str = "rrf",
        bm25_weight: float = 0.4,
        embedding_weight: float = 0.4,
        tfidf_weight: float = 0.2,
        rrf_k: int = 60,
        embedding_model_name: str = "sentence-transformers/msmarco-distilbert-base-v4",
    ) -> None:
        if mode not in ("parallel", "serial"):
            raise ValueError(f"mode must be 'parallel' or 'serial', got '{mode}'.")
        if fusion_method not in ("rrf", "linear"):
            raise ValueError(
                f"fusion_method must be 'rrf' or 'linear', got '{fusion_method}'."
            )

        self.mode = mode
        self.fusion_method = fusion_method
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.tfidf_weight = tfidf_weight
        self.rrf_k = rrf_k

        self.tfidf = TFIDFModel()
        self.bm25 = BM25Model()
        self.embedding = EmbeddingModel(model_name=embedding_model_name)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, documents: List[Dict], tokens_map: Dict[str, List[str]]) -> None:
        """Fit all three underlying models.

        Args:
            documents: List of ``{doc_id, text}`` dicts — used by the
                embedding model (raw text) and to build the token lists
                for TF-IDF and BM25.
            tokens_map: ``{doc_id: [token, ...]}`` mapping of pre-processed
                tokens produced by the preprocessing service.

        Raises:
            ValueError: If any ``doc_id`` in *documents* is absent from
                *tokens_map*.
        """
        missing = [d["doc_id"] for d in documents if d["doc_id"] not in tokens_map]
        if missing:
            raise ValueError(
                f"{len(missing)} doc_id(s) not found in tokens_map: "
                f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
            )

        # Build tokenised doc list for sparse models.
        tokenised_docs = [
            {"doc_id": d["doc_id"], "tokens": tokens_map[d["doc_id"]]}
            for d in documents
        ]

        logger.info("Fitting TF-IDF model…")
        self.tfidf.fit(tokenised_docs)

        logger.info("Fitting BM25 model…")
        self.bm25.fit(tokenised_docs)

        logger.info("Loading embedding model and building FAISS index…")
        self.embedding.load_model()
        self.embedding.build_index(documents)

        logger.info("HybridModel fit complete (%d documents).", len(documents))

    # ------------------------------------------------------------------
    # Serial retrieval (BM25 → embedding re-rank)
    # ------------------------------------------------------------------

    def search_serial(
        self,
        query_text: str,
        query_tokens: List[str],
        top_k: int = 10,
    ) -> List[Dict]:
        """Two-stage pipeline: BM25 candidates → embedding re-rank.

        Step 1: BM25 retrieves ``_CANDIDATE_K`` candidates from the full
                corpus (fast lexical pre-filter).
        Step 2: The embedding model scores only those candidates and
                re-ranks them (expensive dense scoring on a small set).

        Args:
            query_text: Raw query string (used by the embedding model).
            query_tokens: Pre-processed tokens (used by BM25).
            top_k: Final number of results to return.

        Returns:
            List of ``{doc_id, score, rank, bm25_score, embedding_score}``
            dicts sorted by descending embedding score.
        """
        logger.info("[serial] BM25 candidate retrieval (k=%d)…", _CANDIDATE_K)
        bm25_results = self.bm25.search(query_tokens, top_k=_CANDIDATE_K)

        if not bm25_results:
            logger.warning("[serial] BM25 returned no candidates.")
            return []

        bm25_score_map: Dict[str, float] = {
            r["doc_id"]: r["score"] for r in bm25_results
        }
        candidate_ids = [r["doc_id"] for r in bm25_results]

        logger.info("[serial] Re-ranking %d candidates with embeddings…", len(candidate_ids))
        query_vec = self.embedding.encode_query(query_text)

        scored: List[Dict] = []
        for doc_id in candidate_ids:
            try:
                doc_vec = self.embedding.get_embedding_for_doc(doc_id)
                emb_score = float(np.dot(query_vec, doc_vec))
            except (KeyError, RuntimeError):
                emb_score = 0.0
            scored.append(
                {
                    "doc_id": doc_id,
                    "score": emb_score,
                    "bm25_score": bm25_score_map.get(doc_id, 0.0),
                    "embedding_score": emb_score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item["rank"] = rank

        _log_top_results("[serial]", scored[:top_k])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Parallel retrieval (fusion)
    # ------------------------------------------------------------------

    def search_parallel(
        self,
        query_text: str,
        query_tokens: List[str],
        top_k: int = 10,
    ) -> List[Dict]:
        """Three-way parallel retrieval with rank/score fusion.

        All three models score the full corpus independently; their ranked
        lists are merged using either RRF or weighted linear combination.

        Args:
            query_text: Raw query string (embedding model).
            query_tokens: Pre-processed tokens (TF-IDF, BM25).
            top_k: Final number of results to return.

        Returns:
            List of ``{doc_id, score, rank, bm25_score, tfidf_score,
            embedding_score}`` dicts sorted by descending fused score.
        """
        logger.info("[parallel] Retrieving candidates from all three models…")

        bm25_results = self.bm25.search(query_tokens, top_k=_CANDIDATE_K)
        tfidf_results = self.tfidf.search(query_tokens, top_k=_CANDIDATE_K)
        emb_results = self.embedding.search(query_text, top_k=_CANDIDATE_K)

        logger.info(
            "[parallel] Candidates — BM25: %d, TF-IDF: %d, Embedding: %d.",
            len(bm25_results),
            len(tfidf_results),
            len(emb_results),
        )

        # Per-model score maps for provenance logging.
        bm25_map = {r["doc_id"]: r["score"] for r in bm25_results}
        tfidf_map = {r["doc_id"]: r["score"] for r in tfidf_results}
        emb_map = {r["doc_id"]: r["score"] for r in emb_results}

        if self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(
                [bm25_results, tfidf_results, emb_results]
            )
        else:
            fused = self._linear_fusion(
                [bm25_results, tfidf_results, emb_results],
                [self.bm25_weight, self.tfidf_weight, self.embedding_weight],
            )

        # Annotate with per-model scores for transparency.
        for item in fused:
            doc_id = item["doc_id"]
            item["bm25_score"] = bm25_map.get(doc_id, 0.0)
            item["tfidf_score"] = tfidf_map.get(doc_id, 0.0)
            item["embedding_score"] = emb_map.get(doc_id, 0.0)

        top = fused[:top_k]
        _log_top_results("[parallel]", top)
        return top

    # ------------------------------------------------------------------
    # Unified search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        query_tokens: List[str],
        top_k: int = 10,
        mode: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve documents using the configured (or overridden) mode.

        Args:
            query_text: Raw query string.
            query_tokens: Pre-processed query tokens.
            top_k: Number of results to return.
            mode: ``"parallel"`` or ``"serial"``.  Overrides ``self.mode``
                for this call only.

        Returns:
            Ranked list of result dicts.

        Raises:
            ValueError: If *mode* is not ``"parallel"`` or ``"serial"``.
        """
        effective_mode = mode if mode is not None else self.mode
        if effective_mode == "serial":
            return self.search_serial(query_text, query_tokens, top_k)
        if effective_mode == "parallel":
            return self.search_parallel(query_text, query_tokens, top_k)
        raise ValueError(
            f"mode must be 'parallel' or 'serial', got '{effective_mode}'."
        )

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(self, rankings: List[List[Dict]]) -> List[Dict]:
        """Merge ranked lists using Reciprocal Rank Fusion.

        RRF score: ``sum(1 / (k + rank_i(d)))`` across all rankings *i*
        that contain document *d*.  Documents not present in a ranking
        receive no contribution from it (not penalised).

        Args:
            rankings: Each inner list is a ranked result list whose items
                have ``doc_id`` and ``score`` keys.  The list position
                (0-based) is used as the rank.

        Returns:
            Merged list sorted by descending RRF score with ``rank`` set.
        """
        rrf_scores: Dict[str, float] = {}
        for ranked_list in rankings:
            for position, item in enumerate(ranked_list):
                doc_id = item["doc_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    self.rrf_k + position + 1
                )

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"doc_id": doc_id, "score": score, "rank": rank + 1}
            for rank, (doc_id, score) in enumerate(sorted_items)
        ]

    def _linear_fusion(
        self, rankings: List[List[Dict]], weights: List[float]
    ) -> List[Dict]:
        """Merge ranked lists using min-max normalised weighted sum.

        Each ranking's scores are normalised to [0, 1] independently
        before weighting so that models with very different score ranges
        do not dominate.

        Args:
            rankings: Ranked result lists (same format as
                :meth:`_reciprocal_rank_fusion`).
            weights: One weight per ranking; need not sum to 1.

        Returns:
            Merged list sorted by descending fused score with ``rank`` set.
        """
        all_doc_ids: set = set()
        normalised: List[Dict[str, float]] = []

        for ranked_list in rankings:
            norm = _minmax_normalize({r["doc_id"]: r["score"] for r in ranked_list})
            normalised.append(norm)
            all_doc_ids.update(norm.keys())

        fused: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            fused[doc_id] = sum(
                w * norm.get(doc_id, 0.0)
                for w, norm in zip(weights, normalised)
            )

        sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [
            {"doc_id": doc_id, "score": score, "rank": rank + 1}
            for rank, (doc_id, score) in enumerate(sorted_items)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all sub-models and hybrid configuration to *path*.

        The embedding FAISS index is saved to a sub-directory; TF-IDF and
        BM25 are pickled alongside the config.

        Args:
            path: Target directory (created if absent).
        """
        os.makedirs(path, exist_ok=True)

        # Save embedding model (directory-based).
        self.embedding.save(os.path.join(path, "embedding"))

        # Save sparse models.
        self.tfidf.save(os.path.join(path, "tfidf.joblib"))
        self.bm25.save(os.path.join(path, "bm25.pkl"))

        # Save hybrid config.
        config = {
            "mode": self.mode,
            "fusion_method": self.fusion_method,
            "bm25_weight": self.bm25_weight,
            "embedding_weight": self.embedding_weight,
            "tfidf_weight": self.tfidf_weight,
            "rrf_k": self.rrf_k,
        }
        with open(os.path.join(path, "hybrid_config.pkl"), "wb") as fh:
            pickle.dump(config, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("HybridModel saved to '%s'.", path)

    def load(self, path: str) -> None:
        """Load all sub-models and configuration from *path*.

        Args:
            path: Source directory produced by :meth:`save`.

        Raises:
            FileNotFoundError: If required files are absent.
        """
        config_path = os.path.join(path, "hybrid_config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: '{config_path}'")

        with open(config_path, "rb") as fh:
            config = pickle.load(fh)

        self.mode = config["mode"]
        self.fusion_method = config["fusion_method"]
        self.bm25_weight = config["bm25_weight"]
        self.embedding_weight = config["embedding_weight"]
        self.tfidf_weight = config["tfidf_weight"]
        self.rrf_k = config["rrf_k"]

        self.tfidf.load(os.path.join(path, "tfidf.joblib"))
        self.bm25.load(os.path.join(path, "bm25.pkl"))
        self.embedding.load(os.path.join(path, "embedding"))

        logger.info("HybridModel loaded from '%s'.", path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return combined statistics for all sub-models.

        Returns:
            Dict with keys ``mode``, ``fusion_method``, ``weights``,
            ``rrf_k``, ``tfidf``, ``bm25``, ``embedding``.
        """
        return {
            "mode": self.mode,
            "fusion_method": self.fusion_method,
            "weights": {
                "bm25": self.bm25_weight,
                "tfidf": self.tfidf_weight,
                "embedding": self.embedding_weight,
            },
            "rrf_k": self.rrf_k,
            "tfidf": self.tfidf.get_stats(),
            "bm25": self.bm25.get_stats(),
            "embedding": self.embedding.get_stats(),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a score dict to [0, 1].  Returns {} for empty input."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo or 1.0
    return {doc_id: (s - lo) / span for doc_id, s in scores.items()}


def _log_top_results(prefix: str, results: List[Dict], n: int = 5) -> None:
    """Emit a compact debug log of the top-n results."""
    for item in results[:n]:
        parts = [f"doc_id={item['doc_id']}", f"score={item['score']:.4f}"]
        for key in ("bm25_score", "tfidf_score", "embedding_score"):
            if key in item:
                parts.append(f"{key.split('_')[0]}={item[key]:.4f}")
        logger.debug("%s rank=%d  %s", prefix, item.get("rank", "?"), "  ".join(parts))


# Lazy import to avoid circular dependency at module load time.
try:
    import numpy as np
except ImportError as exc:
    raise ImportError("numpy is required for HybridModel serial re-ranking.") from exc