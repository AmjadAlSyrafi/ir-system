"""SQLite-backed document store for full-text retrieval."""

import logging
import os
import sqlite3
from typing import Dict, List

logger = logging.getLogger(__name__)


class DocumentStore:
    """Persistent SQLite store for original document texts.

    Schema: documents(doc_id TEXT, dataset TEXT, text TEXT)
    Primary key is (doc_id, dataset) so multiple datasets can coexist.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id  TEXT NOT NULL,
                    dataset TEXT NOT NULL DEFAULT '',
                    text    TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (doc_id, dataset)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ds ON documents(dataset)"
            )
        logger.info("DocumentStore ready at '%s'.", db_path)

    def store(self, docs: List[Dict], dataset: str = "") -> int:
        """Insert or replace documents.

        Args:
            docs: List of dicts with 'doc_id' and 'text' keys.
            dataset: Dataset tag used to namespace doc_ids.

        Returns:
            Number of documents stored.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO documents (doc_id, dataset, text) VALUES (?, ?, ?)",
                [(d["doc_id"], dataset, d.get("text", "")) for d in docs],
            )
        logger.info("Stored %d documents for dataset='%s'.", len(docs), dataset)
        return len(docs)

    def get_batch(self, doc_ids: List[str], dataset: str = "") -> Dict[str, str]:
        """Fetch texts for a batch of doc_ids.

        Args:
            doc_ids: Document identifiers to look up.
            dataset: Dataset tag that namespaces the doc_ids.

        Returns:
            Dict mapping doc_id → original text. Missing ids are omitted.
        """
        if not doc_ids:
            return {}
        placeholders = ",".join("?" * len(doc_ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT doc_id, text FROM documents "
                f"WHERE dataset=? AND doc_id IN ({placeholders})",
                [dataset, *doc_ids],
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count(self, dataset: str = "") -> int:
        """Return the number of stored documents for a dataset."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE dataset=?", (dataset,)
            ).fetchone()
        return row[0] if row else 0
