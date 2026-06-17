"""
Dense retrieval model using sentence-transformers + FAISS.

Encodes documents into fixed-size embeddings and uses approximate
nearest-neighbour search via FAISS for sub-millisecond retrieval.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"


class EmbeddingModel:
    """Dense retrieval model backed by sentence-transformers and FAISS.

    Args:
        model_name: HuggingFace model identifier.  Defaults to a model
            fine-tuned for passage retrieval (MS MARCO).
        device: ``"cuda"`` or ``"cpu"``.  Auto-detected when ``None``.
        batch_size: Texts to encode per forward pass.
        index_type: ``"flat"`` for exact search (IndexFlatIP) or ``"ivf"``
            for approximate search (IndexIVFFlat).  IVF requires >1 000
            documents to train.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
        index_type: str = "flat",
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.batch_size = batch_size
        self.index_type = index_type

        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._doc_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_dim: int = 0

    # ------------------------------------------------------------------
    # Model / index initialisation
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the sentence-transformer and initialise an empty FAISS index.

        Safe to call multiple times; re-initialises on each call to allow
        device switching without creating a new instance.
        """
        logger.info(
            "Loading model '%s' on device '%s'.", self.model_name, self.device
        )
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d.", self._embedding_dim)
        self._init_faiss_index(self._embedding_dim)

    def _init_faiss_index(self, dim: int) -> None:
        if self.index_type == "ivf":
            quantiser = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(
                quantiser, dim, 100, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self._index = faiss.IndexFlatIP(dim)
        logger.info(
            "Initialised FAISS index type='%s' (dim=%d).", self.index_type, dim
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_documents(
        self, documents: List[Dict], show_progress: bool = True
    ) -> np.ndarray:
        """Encode document texts into L2-normalised embedding vectors.

        Args:
            documents: List of dicts containing at minimum a ``text`` key.
            show_progress: Show a tqdm progress bar over batches.

        Returns:
            Float32 numpy array of shape ``(n_docs, embedding_dim)``.

        Raises:
            RuntimeError: If the model has not been loaded via
                :meth:`load_model`.
        """
        self._require_model()
        texts = [d["text"] for d in documents]
        logger.info(
            "Encoding %d documents (batch_size=%d, device=%s).",
            len(texts),
            self.batch_size,
            self.device,
        )

        batches = range(0, len(texts), self.batch_size)
        iterator = (
            tqdm(batches, desc="Encoding documents", unit="batch")
            if show_progress
            else batches
        )
        all_vecs: List[np.ndarray] = []

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            try:
                vecs = self._model.encode(  # type: ignore[union-attr]
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc):
                    logger.warning(
                        "CUDA OOM on batch starting at %d; retrying on CPU.", start
                    )
                    vecs = self._model.encode(  # type: ignore[union-attr]
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        device="cpu",
                    )
                else:
                    raise
            all_vecs.append(vecs.astype(np.float32))

        return np.vstack(all_vecs)

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a single query string into an L2-normalised vector.

        Args:
            query_text: Raw query string.

        Returns:
            Float32 array of shape ``(embedding_dim,)``.
        """
        self._require_model()
        vec = self._model.encode(  # type: ignore[union-attr]
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, documents: List[Dict]) -> None:
        """Encode all documents and populate the FAISS index.

        For ``index_type="ivf"`` the index must be trained before adding
        vectors; at least 1 000 documents are required for training.
        Fewer documents cause an automatic fallback to a flat index.

        Args:
            documents: List of dicts with ``doc_id`` and ``text`` keys.
        """
        self._require_model()
        if not documents:
            raise ValueError("'documents' must not be empty.")

        self._doc_ids = [d["doc_id"] for d in documents]
        embeddings = self.encode_documents(documents, show_progress=True)
        self._embeddings = embeddings

        assert self._index is not None

        if self.index_type == "ivf":
            if len(documents) < 1000:
                logger.warning(
                    "IVF index requires ≥1000 docs to train; got %d. "
                    "Falling back to flat index.",
                    len(documents),
                )
                self._init_faiss_index(self._embedding_dim)
            else:
                logger.info(
                    "Training IVF index on %d vectors…", len(documents)
                )
                self._index.train(embeddings)  # type: ignore[union-attr]

        self._index.add(embeddings)  # type: ignore[union-attr]
        logger.info(
            "FAISS index built: %d vectors, type='%s'.",
            self._index.ntotal,
            self.index_type,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Return the top-k most similar documents for *query_text*.

        Similarity is inner-product (equivalent to cosine similarity for
        L2-normalised vectors).

        Args:
            query_text: Raw query string.
            top_k: Number of results to return.

        Returns:
            List of ``{doc_id, score, rank}`` dicts sorted by descending
            score.
        """
        self._require_index()
        query_vec = self.encode_query(query_text).reshape(1, -1)

        k = min(top_k, len(self._doc_ids))
        scores, indices = self._index.search(query_vec, k)  # type: ignore[union-attr]

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            results.append(
                {
                    "doc_id": self._doc_ids[int(idx)],
                    "score": float(score),
                    "rank": rank + 1,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Embedding retrieval
    # ------------------------------------------------------------------

    def get_embedding_for_doc(self, doc_id: str) -> np.ndarray:
        """Return the stored embedding vector for *doc_id*.

        Useful for hybrid model score fusion.

        Args:
            doc_id: Document identifier.

        Returns:
            Float32 array of shape ``(embedding_dim,)``.

        Raises:
            KeyError: If *doc_id* is not in the index.
            RuntimeError: If embeddings were not retained.
        """
        if doc_id not in self._doc_ids:
            raise KeyError(f"doc_id='{doc_id}' not found in index.")
        if self._embeddings is None:
            raise RuntimeError(
                "Embeddings not available. Rebuild the index or re-save "
                "with embeddings retained."
            )
        idx = self._doc_ids.index(doc_id)
        return self._embeddings[idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the FAISS index and metadata to the directory *path*.

        The FAISS index is written with ``faiss.write_index``; doc_id
        mapping and embeddings are pickled alongside it.

        Args:
            path: Target directory (created if absent).
        """
        self._require_index()
        os.makedirs(path, exist_ok=True)

        faiss.write_index(  # type: ignore[union-attr]
            self._index, os.path.join(path, "faiss.index")
        )

        meta = {
            "doc_ids": self._doc_ids,
            "embeddings": self._embeddings,
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "index_type": self.index_type,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as fh:
            pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("EmbeddingModel saved to '%s'.", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from directory *path*.

        The sentence-transformer itself is not stored; call
        :meth:`load_model` separately if you need to encode new texts.

        Args:
            path: Source directory.

        Raises:
            FileNotFoundError: If expected files are missing.
        """
        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "meta.pkl")
        for p in (index_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Expected file not found: '{p}'")

        self._index = faiss.read_index(index_path)

        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)

        self._doc_ids = meta["doc_ids"]
        self._embeddings = meta.get("embeddings")
        self.model_name = meta.get("model_name", self.model_name)
        self._embedding_dim = meta.get("embedding_dim", 0)
        self.index_type = meta.get("index_type", self.index_type)

        logger.info(
            "EmbeddingModel loaded from '%s': %d docs.", path, len(self._doc_ids)
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics.

        Returns:
            Dict with keys ``num_documents``, ``embedding_dim``,
            ``model_name``, ``index_type``.
        """
        return {
            "num_documents": len(self._doc_ids),
            "embedding_dim": self._embedding_dim,
            "model_name": self.model_name,
            "index_type": self.index_type,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

    def _require_index(self) -> None:
        if self._index is None or len(self._doc_ids) == 0:
            raise RuntimeError(
                "FAISS index is empty. Call build_index() or load() first."
            )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False